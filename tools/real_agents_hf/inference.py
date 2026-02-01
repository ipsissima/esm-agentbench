#!/usr/bin/env python3
"""Local inference backend for Hugging Face models.

Supports both transformers and vLLM backends with automatic fallback.
All operations are local - no external API calls.

ESM_STRICT mode:
    Set ESM_STRICT=1 to enable strict invariant checking. This will:
    - Verify all attention modules use the same implementation post-load
    - Fail fast on configuration mismatches instead of silently continuing
"""
from __future__ import annotations

import importlib.util
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Apply transformers compatibility shims early (before any model loading)
from .shims import apply_transformers_shims
apply_transformers_shims()

# Import runtime policy for centralized environment decisions
from .runtime_policy import get_runtime_policy

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a local model."""
    name: str
    hf_id: str
    backend: str
    dtype: str
    max_tokens: int
    temperature: float
    context_length: int = 2048  # Default context window size
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    description: str = ""
    revision: Optional[str] = None  # HF model revision/commit SHA for deterministic loads

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        return cls(
            name=data['name'],
            hf_id=data['hf_id'],
            backend=data['backend'],
            dtype=data['dtype'],
            max_tokens=data['max_tokens'],
            temperature=data['temperature'],
            context_length=data.get('context_length', 2048),
            load_in_4bit=data.get('load_in_4bit', False),
            load_in_8bit=data.get('load_in_8bit', False),
            description=data.get('description', ''),
            revision=data.get('revision'),
        )


def load_model_configs(config_path: Optional[Path] = None) -> List[ModelConfig]:
    """Load model configurations from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "models.yaml"

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return [ModelConfig.from_dict(m) for m in data['models']]


def get_model_config(model_name: str, config_path: Optional[Path] = None) -> ModelConfig:
    """Get configuration for a specific model."""
    configs = load_model_configs(config_path)
    for cfg in configs:
        if cfg.name == model_name:
            return cfg
    raise ValueError(f"Model {model_name} not found in config")


class InferenceBackend:
    """Base class for inference backends."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        # If we detect incompatibilities with transformers cache internals,
        # this flag forces generation calls to disable use_cache (safer).
        self._force_no_cache = False

    def load(self):
        """Load model and tokenizer."""
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, stop: Optional[List[str]] = None) -> str:
        """Generate text from prompt."""
        raise NotImplementedError

    def unload(self):
        """Unload model to free memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class TransformersBackend(InferenceBackend):
    """Inference backend using Hugging Face transformers."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Residual stream instrumentation
        self._residual_hooks: List[Any] = []
        self._residual_stream: List[np.ndarray] = []
        self._capture_residuals = os.getenv("ESM_CAPTURE_INTERNAL", "0") == "1"

    def _register_residual_hooks(self, layers: Optional[List[str]] = None) -> None:
        """Register forward hooks to capture residual stream.

        Parameters
        ----------
        layers : Optional[List[str]]
            List of layer names to capture. If None, captures all transformer layers.
        """
        if self.model is None:
            return

        self._residual_stream = []

        def make_hook(layer_name: str):
            def hook(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                # Convert to numpy and take mean over sequence
                if hasattr(hidden, "detach"):
                    arr = hidden[:, -1, :].detach().cpu().numpy()
                    self._residual_stream.append(arr.squeeze())
                elif isinstance(hidden, np.ndarray):
                    arr = hidden[:, -1, :]
                    self._residual_stream.append(arr.squeeze())
                elif hasattr(hidden, "numpy"):
                    arr = hidden[:, -1, :].numpy()
                    self._residual_stream.append(arr.squeeze())
            return hook

        # Find transformer layers
        layer_keywords = ("transformer", "decoder", "encoder", "layer", "block", "h.")
        if layers is None:
            layers = []
            for name, module in self.model.named_modules():
                name_lower = name.lower()
                # Match common transformer layer patterns
                if any(kw in name_lower for kw in layer_keywords):
                    # Avoid matching sub-components
                    if not any(x in name_lower for x in ("attn", "mlp", "norm", "ln")):
                        layers.append(name)

        # Register hooks
        for name, module in self.model.named_modules():
            if name in layers or (layers and any(l in name for l in layers)):
                handle = module.register_forward_hook(make_hook(name))
                self._residual_hooks.append(handle)
                logger.debug(f"Registered residual hook on: {name}")

    def _remove_residual_hooks(self) -> None:
        """Remove all registered residual hooks."""
        for handle in self._residual_hooks:
            handle.remove()
        self._residual_hooks = []

    def get_residual_stream(self) -> List[np.ndarray]:
        """Get captured residual stream from last generation.

        Returns
        -------
        List[np.ndarray]
            List of hidden state arrays, one per layer.
        """
        return self._residual_stream.copy()

    def clear_residual_stream(self) -> None:
        """Clear captured residual stream."""
        self._residual_stream = []

    def load(self):
        """Load model using transformers.

        This method uses RuntimePolicy for CPU-safe loading logic:
        - Disables 4/8-bit quantization on CPU (requires bitsandbytes + CUDA)
        - Forces float32 dtype on CPU to avoid tensor shape mismatches
        - Forces eager attention when flash-attn is missing
        - Pins model revision when specified for deterministic loads
        - Forces use_cache=False on CPU to avoid past-key-value compatibility issues
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: "
                "pip install transformers accelerate"
            )

        logger.info(f"Loading {self.config.hf_id} with transformers backend...")

        # Get runtime policy for centralized environment decisions
        policy = get_runtime_policy()
        logger.debug("RuntimePolicy: %s", policy)

        # --- Apply policy-based safety rules ---
        # Disable quantization if not allowed
        if not policy.allow_quant():
            if getattr(self.config, "load_in_4bit", False) or getattr(self.config, "load_in_8bit", False):
                logger.warning(
                    "Quantization requested but not allowed by policy (cuda=%s, bnb=%s). Disabling.",
                    policy.cuda, policy.have_bnb
                )
            self.config.load_in_4bit = False
            self.config.load_in_8bit = False

        # Force safe dtype based on policy
        recommended_dtype = policy.recommended_dtype(getattr(self.config, "dtype", None))
        if recommended_dtype != getattr(self.config, "dtype", None):
            logger.info("Policy recommends dtype=%s (was %s)", recommended_dtype, self.config.dtype)
            self.config.dtype = recommended_dtype

        # Force no-cache mode if policy disallows caching
        if not policy.allow_cache():
            self._force_no_cache = True
            logger.info("Policy: forcing use_cache=False for safe generation")
        # --- end policy-based safety ---

        # Prepare kwargs using policy helpers
        model_kwargs = policy.get_model_kwargs(
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            dtype=self.config.dtype,
        )

        # Pin model revision if specified for deterministic loads
        revision = getattr(self.config, "revision", None)
        tokenizer_kwargs = {"trust_remote_code": True}
        if revision:
            model_kwargs["revision"] = revision
            tokenizer_kwargs["revision"] = revision
            logger.info(f"Using pinned model revision: {revision}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_id,
            **tokenizer_kwargs,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_id,
            **model_kwargs,
        )

        # Post-load: sync attn_implementation on model config for remote code that checks it
        # This ensures consistent behavior even if remote model code reads config after loading
        attn_impl = policy.attn_impl()
        try:
            if hasattr(self.model, "config"):
                self.model.config.attn_implementation = attn_impl
                # Also set _attn_implementation for internal use (private transformers attribute)
                if hasattr(self.model.config, "_attn_implementation"):
                    self.model.config._attn_implementation = attn_impl
                # Some models use attention dict
                if hasattr(self.model.config, "attention") and isinstance(self.model.config.attention, dict):
                    self.model.config.attention["implementation"] = attn_impl
                # Some models use _attn_implementation_internal
                if hasattr(self.model.config, "_attn_implementation_internal"):
                    self.model.config._attn_implementation_internal = attn_impl
                logger.debug("Synced model.config.attn_implementation = %s", attn_impl)
        except Exception as e:
            logger.debug(f"Could not sync attn_implementation on model config: {e}")

        # ESM_STRICT mode: verify attention contract is satisfied
        # Invariant: All attention submodules must agree on attention implementation
        if os.getenv("ESM_STRICT", "0") == "1":
            mismatches = self._verify_attention_contract(attn_impl)
            if mismatches:
                raise RuntimeError(
                    f"Attention implementation mismatch post-load (ESM_STRICT=1): {mismatches}. "
                    f"Expected all modules to use '{attn_impl}'."
                )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model {self.config.name} loaded successfully")

    def _verify_attention_contract(self, expected_impl: str) -> List[tuple]:
        """Verify all attention modules agree on implementation.

        Checks attention-related submodules for consistency with the expected
        implementation. This enforces the invariant that all attention modules
        must use the same implementation to avoid shape mismatches.

        Parameters
        ----------
        expected_impl : str
            The expected attention implementation ("eager", "flash_attention_2", "sdpa").

        Returns
        -------
        List[tuple]
            List of (module_name, found_impl) tuples for mismatching modules.
            Empty list if all modules match.
        """
        if self.model is None:
            return []

        mismatches = []
        attention_keywords = ("attn", "attention", "self_attn", "multihead")

        for name, module in self.model.named_modules():
            # Heuristically identify attention-like modules
            name_lower = name.lower()
            if not any(k in name_lower for k in attention_keywords):
                continue

            # Check various places where attention implementation might be stored
            found_impl = None

            # Check direct attribute
            if hasattr(module, "attn_impl"):
                found_impl = getattr(module, "attn_impl", None)
            elif hasattr(module, "_attn_implementation"):
                found_impl = getattr(module, "_attn_implementation", None)
            elif hasattr(module, "implementation"):
                found_impl = getattr(module, "implementation", None)

            # Check module config if present
            if found_impl is None and hasattr(module, "config"):
                cfg = module.config
                if hasattr(cfg, "attn_implementation"):
                    found_impl = cfg.attn_implementation
                elif hasattr(cfg, "_attn_implementation"):
                    found_impl = cfg._attn_implementation
                elif isinstance(getattr(cfg, "attention", None), dict):
                    found_impl = cfg.attention.get("implementation")

            # Only flag if we found a conflicting implementation
            if found_impl is not None and found_impl != expected_impl:
                mismatches.append((name, found_impl))

        return mismatches

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, stop: Optional[List[str]] = None,
                 capture_residuals: Optional[bool] = None) -> str:
        """Generate text from prompt using transformers.

        Parameters
        ----------
        prompt : str
            Input prompt.
        max_tokens : Optional[int]
            Maximum tokens to generate.
        temperature : Optional[float]
            Sampling temperature.
        stop : Optional[List[str]]
            Stop sequences.
        capture_residuals : Optional[bool]
            If True, capture residual stream during generation.
            If None, uses ESM_CAPTURE_INTERNAL env var.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Determine if we should capture residuals
        should_capture = capture_residuals if capture_residuals is not None else self._capture_residuals
        if should_capture:
            self.clear_residual_stream()
            self._register_residual_hooks()

        # Apply CPU-specific token limit (without KV cache, generation is O(n²))
        policy = get_runtime_policy()
        max_tokens = policy.max_tokens_for_cpu(max_tokens)

        # Calculate max input length, leaving room for generation
        max_input_length = self.config.context_length - max_tokens

        # Tokenize with truncation to prevent exceeding model's context window
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate — try normal generation, but be defensive for cache-internal
        # AttributeErrors coming from remote model code mismatching transformers runtime.
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # If we've seen cache incompatibilities before, force no cache.
        if getattr(self, "_force_no_cache", False):
            gen_kwargs["use_cache"] = False

        with torch.no_grad():
            try:
                outputs = self.model.generate(**gen_kwargs)
            except AttributeError as ex:
                # Common symptoms: remote model code expects DynamicCache.seen_tokens,
                # DynamicCache.get_max_length, or other cache internals not present.
                msg = str(ex)
                if any(x in msg for x in ("seen_tokens", "get_max_length", "get_usable_length", "DynamicCache")):
                    logger.warning(
                        "Model generation failed due to cache-internals mismatch (%s). "
                        "Retrying with use_cache=False (slower).", msg
                    )
                    # Persist the decision so subsequent generates avoid the slow fail/retry.
                    self._force_no_cache = True
                    gen_kwargs["use_cache"] = False
                    outputs = self.model.generate(**gen_kwargs)
                else:
                    # Unknown AttributeError - re-raise so it surfaces
                    raise
            except RuntimeError as ex:
                # Handle tensor shape mismatches that can occur with mixed attention paths
                # or cache incompatibilities (e.g., 970 vs 1940 shape errors)
                msg = str(ex)
                if any(x in msg.lower() for x in ("shape", "size", "mismatch", "dimension")):
                    if not getattr(self, "_force_no_cache", False):
                        logger.warning(
                            "Model generation failed due to tensor shape mismatch (%s). "
                            "Retrying with use_cache=False.", msg[:200]
                        )
                        self._force_no_cache = True
                        gen_kwargs["use_cache"] = False
                        outputs = self.model.generate(**gen_kwargs)
                    else:
                        # Already tried with no cache, re-raise
                        logger.error(
                            "Tensor shape mismatch persists even with use_cache=False. "
                            "This may indicate incompatible model code. Run debug_phi3_shapes.py for diagnosis."
                        )
                        raise
                else:
                    # Unknown RuntimeError - re-raise
                    raise

        # Decode (skip input prompt)
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in generated:
                    generated = generated[:generated.index(stop_seq)]

        # Clean up residual hooks
        if should_capture:
            self._remove_residual_hooks()

        return generated.strip()


class VLLMBackend(InferenceBackend):
    """Inference backend using vLLM (much faster for multiple runs)."""

    def load(self):
        """Load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Note: vLLM requires CUDA and is Linux-only."
            )

        logger.info(f"Loading {self.config.hf_id} with vLLM backend...")

        # Prepare kwargs
        model_kwargs = {
            "model": self.config.hf_id,
            "trust_remote_code": True,
            "dtype": self.config.dtype,
        }

        # vLLM automatically manages GPU memory and batching
        self.model = LLM(**model_kwargs)

        logger.info(f"Model {self.config.name} loaded successfully with vLLM")

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, stop: Optional[List[str]] = None) -> str:
        """Generate text from prompt using vLLM."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from vllm import SamplingParams

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()


def create_backend(model_name: str, config_path: Optional[Path] = None) -> InferenceBackend:
    """Create appropriate inference backend for the model.

    Parameters
    ----------
    model_name : str
        Name of the model from models.yaml
    config_path : Path, optional
        Path to models.yaml config file

    Returns
    -------
    InferenceBackend
        Configured backend (vLLM or transformers)
    """
    config = get_model_config(model_name, config_path)

    if config.backend == "vllm":
        try:
            backend = VLLMBackend(config)
            logger.info(f"Using vLLM backend for {model_name}")
            return backend
        except ImportError:
            logger.warning(
                f"vLLM not available, falling back to transformers for {model_name}"
            )
            config.backend = "transformers"

    if config.backend == "transformers":
        return TransformersBackend(config)

    raise ValueError(f"Unknown backend: {config.backend}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with tiny model
    backend = create_backend("tiny-test")
    backend.load()

    prompt = "def fibonacci(n):\n    "
    result = backend.generate(prompt, max_tokens=100)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")

    backend.unload()
