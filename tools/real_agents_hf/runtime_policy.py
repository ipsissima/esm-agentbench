# tools/real_agents_hf/runtime_policy.py
"""Runtime Policy for model loading and inference decisions.

Provides a centralized decision point for:
- Whether to use quantization (requires bitsandbytes + CUDA)
- Which attention implementation to use (flash vs eager)
- Whether to use KV cache during generation
- What dtype to use for model weights

This eliminates scattered if/else checks throughout the codebase and
ensures consistent behavior based on the runtime environment.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Thread-safe initialization lock
_policy_lock = threading.Lock()


@dataclass
class RuntimePolicy:
    """Centralized policy for runtime decisions.

    This class detects the available hardware and software at initialization
    and provides methods to query what code paths should be used. All other
    code should use this policy rather than checking torch.cuda.is_available()
    or importlib.util.find_spec() directly.

    Attributes
    ----------
    cuda : bool
        Whether CUDA is available.
    have_bnb : bool
        Whether bitsandbytes is available.
    have_flash : bool
        Whether flash_attn is available.
    have_accelerate : bool
        Whether accelerate is available.
    """

    cuda: bool
    have_bnb: bool
    have_flash: bool
    have_accelerate: bool

    def __init__(self):
        """Detect runtime environment and set policy flags."""
        # Check CUDA availability
        try:
            import torch
            self.cuda = torch.cuda.is_available()
        except ImportError:
            self.cuda = False

        # Check optional packages
        self.have_bnb = importlib.util.find_spec("bitsandbytes") is not None
        self.have_flash = importlib.util.find_spec("flash_attn") is not None
        self.have_accelerate = importlib.util.find_spec("accelerate") is not None

        logger.debug(
            "RuntimePolicy initialized: cuda=%s, bnb=%s, flash=%s, accelerate=%s",
            self.cuda, self.have_bnb, self.have_flash, self.have_accelerate
        )

    def allow_quant(self) -> bool:
        """Return True if quantization (4-bit/8-bit) is allowed.

        Quantization requires both CUDA and bitsandbytes to be available.
        """
        return self.cuda and self.have_bnb

    def allow_4bit(self) -> bool:
        """Return True if 4-bit quantization is allowed."""
        return self.allow_quant()

    def allow_8bit(self) -> bool:
        """Return True if 8-bit quantization is allowed."""
        return self.allow_quant()

    def attn_impl(self) -> Literal["flash_attention_2", "eager", "sdpa"]:
        """Return the recommended attention implementation.

        Returns normalized names that transformers expects:
        - "flash_attention_2": If CUDA and flash_attn are available
        - "sdpa": If CUDA is available, flash_attn is not, and PyTorch 2.0+ SDPA works
        - "eager": If running on CPU or no optimized attention available (safest)

        Note: We default to "eager" rather than "sdpa" on GPU without flash-attn
        because eager is the most compatible and predictable code path. SDPA can
        cause shape mismatches in some models.
        """
        if self.cuda and self.have_flash:
            return "flash_attention_2"
        elif self.cuda:
            # Check if PyTorch 2.0+ SDPA is available
            try:
                import torch
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    # SDPA available, but we still prefer eager for safety
                    # Uncomment line below to use SDPA when flash-attn unavailable:
                    # return "sdpa"
                    pass
            except Exception:
                pass
            # Default to eager for maximum compatibility
            return "eager"
        else:
            return "eager"

    def allow_cache(self) -> bool:
        """Return True if KV caching during generation is safe.

        Caching past_key_values can cause tensor shape mismatches on CPU
        due to inconsistent cache handling in some models. We only enable
        caching on GPU where the behavior is more tested.
        """
        return self.cuda

    def allow_device_map(self) -> bool:
        """Return True if device_map='auto' is safe to use.

        device_map requires accelerate and CUDA to work properly.
        """
        return self.cuda and self.have_accelerate

    def recommended_dtype(self, requested_dtype: Optional[str] = None) -> str:
        """Return the recommended dtype for model loading.

        Parameters
        ----------
        requested_dtype : str, optional
            The dtype requested in the model config (e.g., "float16", "bfloat16").

        Returns
        -------
        str
            The dtype to actually use ("float32", "float16", or "bfloat16").
        """
        # On CPU, always use float32 to avoid tensor shape mismatches
        if not self.cuda:
            if requested_dtype in ("float16", "bfloat16"):
                logger.warning(
                    "Model requested dtype=%s but running on CPU; forcing float32",
                    requested_dtype
                )
            return "float32"

        # On GPU, use the requested dtype or default to float32
        if requested_dtype in ("float16", "bfloat16", "float32"):
            return requested_dtype

        return "float32"

    def get_model_kwargs(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        dtype: Optional[str] = None,
    ) -> dict:
        """Build model loading kwargs based on policy.

        This method returns a dictionary of kwargs suitable for passing to
        AutoModelForCausalLM.from_pretrained().

        Parameters
        ----------
        load_in_4bit : bool
            Whether 4-bit quantization was requested.
        load_in_8bit : bool
            Whether 8-bit quantization was requested.
        dtype : str, optional
            The dtype requested in the config.

        Returns
        -------
        dict
            Kwargs for model loading.
        """
        import torch

        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Handle device_map
        if self.allow_device_map():
            kwargs["device_map"] = "auto"

        # Handle dtype
        recommended = self.recommended_dtype(dtype)
        if recommended == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif recommended == "float16":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        # Handle quantization
        if load_in_4bit and self.allow_4bit():
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, skipping quantization")
        elif load_in_8bit and self.allow_8bit():
            kwargs["load_in_8bit"] = True
            logger.info("Using 8-bit quantization")
        elif (load_in_4bit or load_in_8bit) and not self.allow_quant():
            logger.warning(
                "Quantization requested but not available (cuda=%s, bnb=%s); disabled",
                self.cuda, self.have_bnb
            )

        # Handle attention implementation
        attn = self.attn_impl()
        kwargs["attn_implementation"] = attn
        logger.info("Using attention implementation: %s", attn)

        return kwargs

    def get_generate_kwargs(self) -> dict:
        """Build generation kwargs based on policy.

        Returns
        -------
        dict
            Kwargs for model.generate().
        """
        kwargs = {}

        if not self.allow_cache():
            kwargs["use_cache"] = False
            logger.debug("Disabling KV cache for safe CPU generation")

        return kwargs

    def max_tokens_for_cpu(self, requested_max_tokens: int) -> int:
        """Return the effective max_tokens for CPU execution.

        On CPU without KV cache, each token generation requires recomputing
        all attention weights from scratch. This makes large generation
        budgets prohibitively slow. This method limits max_tokens to a
        reasonable value for CPU execution.

        Parameters
        ----------
        requested_max_tokens : int
            The max_tokens from model config.

        Returns
        -------
        int
            The effective max_tokens to use.
        """
        # CPU generation limit: 256 tokens is reasonable (use_cache=False is O(n²))
        CPU_MAX_TOKENS = 256

        if self.cuda:
            return requested_max_tokens

        if requested_max_tokens > CPU_MAX_TOKENS:
            logger.warning(
                "max_tokens=%d is too slow for CPU (use_cache=False); limiting to %d",
                requested_max_tokens, CPU_MAX_TOKENS
            )
            return CPU_MAX_TOKENS

        return requested_max_tokens

    def __str__(self) -> str:
        return (
            f"RuntimePolicy(cuda={self.cuda}, bnb={self.have_bnb}, "
            f"flash={self.have_flash}, accelerate={self.have_accelerate})"
        )

    def __repr__(self) -> str:
        return self.__str__()


# Global singleton instance (lazy initialized)
_policy: Optional[RuntimePolicy] = None


def get_runtime_policy() -> RuntimePolicy:
    """Get the global RuntimePolicy instance.

    This function is thread-safe and uses double-check locking for
    efficient concurrent access.

    Returns
    -------
    RuntimePolicy
        The global policy instance.
    """
    global _policy

    # Fast path: return cached policy without lock
    if _policy is not None:
        return _policy

    # Slow path: acquire lock for initialization
    with _policy_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if _policy is not None:
            return _policy

        _policy = RuntimePolicy()
        return _policy


def reset_runtime_policy() -> None:
    """Reset the global RuntimePolicy (mainly for testing).

    Thread-safe: acquires lock before clearing.
    """
    global _policy
    with _policy_lock:
        _policy = None


__all__ = [
    "RuntimePolicy",
    "get_runtime_policy",
    "reset_runtime_policy",
]
