"""Model Adapter with Invariant Enforcement.

This module provides a centralized entry point for all model loading and inference,
enforcing architectural invariants that prevent crashes and shape mismatches.

Key Invariants Enforced:
1. Attention invariant: All attention modules use the same implementation
2. Cache contract invariant: KV cache semantics are validated before use
3. Dtype invariant: Model dtype matches policy recommendations
4. Config invariant: Model config is synced with runtime policy

Usage:
    adapter = ModelAdapter.load("microsoft/Phi-3-mini-4k-instruct")
    result = adapter.generate("Hello, world!")

    # Or with compatibility check
    adapter = ModelAdapter.load(model_id, strict=True)
    report = adapter.compatibility_check()
    if not report.is_compatible:
        raise RuntimeError(f"Model incompatible: {report.issues}")
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .shims import apply_transformers_shims
from .runtime_policy import RuntimePolicy, get_runtime_policy

logger = logging.getLogger(__name__)

# Apply shims early
apply_transformers_shims()


class InvariantViolation(RuntimeError):
    """Raised when a model invariant is violated."""


class AttentionImplementation(Enum):
    """Attention implementation types."""
    EAGER = "eager"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    UNKNOWN = "unknown"


@dataclass
class AttentionInfo:
    """Information about an attention module."""
    name: str
    implementation: AttentionImplementation
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    num_kv_heads: Optional[int] = None


@dataclass
class CacheContractInfo:
    """Information about cache contract compliance."""
    has_get_usable_length: bool = False
    has_get_seq_length: bool = False
    has_get_max_length: bool = False
    has_seen_tokens: bool = False
    is_compliant: bool = False


@dataclass
class CompatibilityReport:
    """Report on model compatibility with runtime policy."""
    is_compatible: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    attention_modules: List[AttentionInfo] = field(default_factory=list)
    cache_contract: Optional[CacheContractInfo] = None
    policy: Optional[RuntimePolicy] = None
    config_synced: bool = False

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.is_compatible = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class ModelAdapter:
    """Centralized model adapter with invariant enforcement.

    This class is the single entry point for loading and using models.
    It enforces all architectural invariants and provides a clean interface.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        policy: RuntimePolicy,
        config: Dict[str, Any],
    ):
        """Initialize adapter (use ModelAdapter.load() instead)."""
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy
        self.config = config
        self._force_no_cache = not policy.allow_cache()
        self._compatibility_report: Optional[CompatibilityReport] = None

    @classmethod
    def load(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        strict: bool = False,
        trust_remote_code: bool = True,
    ) -> "ModelAdapter":
        """Load a model with invariant enforcement.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID.
        revision : str, optional
            Model revision/commit SHA for deterministic loads.
        dtype : str, optional
            Requested dtype (float32, float16, bfloat16).
        load_in_4bit : bool
            Use 4-bit quantization.
        load_in_8bit : bool
            Use 8-bit quantization.
        strict : bool
            If True, raise on any invariant violation.
        trust_remote_code : bool
            Whether to trust remote model code.

        Returns
        -------
        ModelAdapter
            Configured model adapter.

        Raises
        ------
        InvariantViolation
            If strict=True and invariants are violated.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_id}")

        # Get runtime policy
        policy = get_runtime_policy()
        logger.debug(f"Runtime policy: {policy}")

        # Apply policy-based overrides
        actual_4bit = load_in_4bit and policy.allow_4bit()
        actual_8bit = load_in_8bit and policy.allow_8bit()
        actual_dtype = policy.recommended_dtype(dtype)
        actual_attn = policy.attn_impl()

        if load_in_4bit and not actual_4bit:
            logger.warning("4-bit quantization not available, disabling")
        if load_in_8bit and not actual_8bit:
            logger.warning("8-bit quantization not available, disabling")

        # Build model kwargs
        model_kwargs = policy.get_model_kwargs(
            load_in_4bit=actual_4bit,
            load_in_8bit=actual_8bit,
            dtype=actual_dtype,
        )
        model_kwargs["trust_remote_code"] = trust_remote_code

        if revision:
            model_kwargs["revision"] = revision
            logger.info(f"Using pinned revision: {revision}")

        # Build tokenizer kwargs
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if revision:
            tokenizer_kwargs["revision"] = revision

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Sync attention implementation on model config
        cls._sync_attn_implementation(model, actual_attn)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create adapter
        config = {
            "model_id": model_id,
            "revision": revision,
            "dtype": actual_dtype,
            "load_in_4bit": actual_4bit,
            "load_in_8bit": actual_8bit,
            "attn_implementation": actual_attn,
        }
        adapter = cls(model, tokenizer, policy, config)

        # Run compatibility check
        report = adapter.compatibility_check()
        adapter._compatibility_report = report

        if strict and not report.is_compatible:
            raise InvariantViolation(
                f"Model incompatible with policy: {report.issues}"
            )

        for warning in report.warnings:
            logger.warning(warning)

        logger.info(f"Model loaded: {model_id} (compatible={report.is_compatible})")
        return adapter

    @staticmethod
    def _sync_attn_implementation(model: Any, attn_impl: str) -> None:
        """Sync attention implementation across model config."""
        if not hasattr(model, "config"):
            return

        try:
            model.config.attn_implementation = attn_impl

            # Also set private attribute used by some models
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = attn_impl

            # Some models use attention dict
            if hasattr(model.config, "attention") and isinstance(
                model.config.attention, dict
            ):
                model.config.attention["implementation"] = attn_impl

            # Some models use _attn_implementation_internal
            if hasattr(model.config, "_attn_implementation_internal"):
                model.config._attn_implementation_internal = attn_impl

            logger.debug(f"Synced model.config.attn_implementation = {attn_impl}")
        except Exception as exc:
            logger.debug(f"Could not sync attn_implementation: {exc}")

    def compatibility_check(self) -> CompatibilityReport:
        """Run comprehensive compatibility check.

        Checks all invariants and returns a detailed report.

        Returns
        -------
        CompatibilityReport
            Report with compatibility status and any issues.
        """
        report = CompatibilityReport(is_compatible=True, policy=self.policy)

        # Check 1: Attention invariant
        self._check_attention_invariant(report)

        # Check 2: Cache contract invariant
        self._check_cache_contract(report)

        # Check 3: Config sync
        self._check_config_sync(report)

        return report

    def _check_attention_invariant(self, report: CompatibilityReport) -> None:
        """Check that all attention modules agree on implementation.

        The attention invariant requires that all attention modules
        use the same implementation to avoid shape mismatches.
        """
        expected_impl = self.config.get("attn_implementation", "eager")
        attention_keywords = ("attn", "attention", "self_attn", "multihead")
        implementations_found: Set[str] = set()

        for name, module in self.model.named_modules():
            name_lower = name.lower()
            if not any(k in name_lower for k in attention_keywords):
                continue

            # Extract attention info
            info = self._extract_attention_info(name, module)
            report.attention_modules.append(info)

            if info.implementation != AttentionImplementation.UNKNOWN:
                implementations_found.add(info.implementation.value)

        # Check for mismatches
        if len(implementations_found) > 1:
            report.add_issue(
                f"Attention implementation mismatch: found {implementations_found}, "
                f"expected all '{expected_impl}'"
            )
        elif implementations_found and expected_impl not in implementations_found:
            # Only one impl found but it's different from expected
            found = list(implementations_found)[0]
            if found != expected_impl:
                report.add_warning(
                    f"Attention implementation differs from policy: "
                    f"found '{found}', policy recommends '{expected_impl}'"
                )

    def _extract_attention_info(self, name: str, module: Any) -> AttentionInfo:
        """Extract attention information from a module."""
        impl = AttentionImplementation.UNKNOWN
        num_heads = None
        head_dim = None
        num_kv_heads = None

        # Check various places for implementation
        for attr in ("attn_impl", "_attn_implementation", "implementation"):
            if hasattr(module, attr):
                impl_str = getattr(module, attr, None)
                if impl_str:
                    impl = self._parse_attn_impl(impl_str)
                    break

        # Check module config
        if impl == AttentionImplementation.UNKNOWN and hasattr(module, "config"):
            cfg = module.config
            for attr in ("attn_implementation", "_attn_implementation"):
                if hasattr(cfg, attr):
                    impl_str = getattr(cfg, attr, None)
                    if impl_str:
                        impl = self._parse_attn_impl(impl_str)
                        break
            if isinstance(getattr(cfg, "attention", None), dict):
                impl_str = cfg.attention.get("implementation")
                if impl_str:
                    impl = self._parse_attn_impl(impl_str)

        # Extract head dimensions
        for attr in ("num_heads", "num_attention_heads", "n_heads"):
            if hasattr(module, attr):
                num_heads = getattr(module, attr, None)
                break

        for attr in ("head_dim", "head_size"):
            if hasattr(module, attr):
                head_dim = getattr(module, attr, None)
                break

        for attr in ("num_kv_heads", "num_key_value_heads"):
            if hasattr(module, attr):
                num_kv_heads = getattr(module, attr, None)
                break

        return AttentionInfo(
            name=name,
            implementation=impl,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )

    def _parse_attn_impl(self, impl_str: str) -> AttentionImplementation:
        """Parse attention implementation string."""
        if not isinstance(impl_str, str):
            return AttentionImplementation.UNKNOWN

        impl_str = impl_str.lower()
        if "flash" in impl_str:
            return AttentionImplementation.FLASH_ATTENTION_2
        elif "sdpa" in impl_str:
            return AttentionImplementation.SDPA
        elif "eager" in impl_str:
            return AttentionImplementation.EAGER
        return AttentionImplementation.UNKNOWN

    def _check_cache_contract(self, report: CompatibilityReport) -> None:
        """Check cache contract compliance.

        The cache contract requires DynamicCache to have specific methods
        for compatibility with model code.
        """
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            report.add_warning("Could not import DynamicCache for cache contract check")
            return

        cache_info = CacheContractInfo()

        # Check required methods
        cache_info.has_get_usable_length = hasattr(DynamicCache, "get_usable_length")
        cache_info.has_get_seq_length = hasattr(DynamicCache, "get_seq_length")
        cache_info.has_get_max_length = hasattr(DynamicCache, "get_max_length")
        cache_info.has_seen_tokens = hasattr(DynamicCache, "seen_tokens")

        # Cache is compliant if all required methods exist
        cache_info.is_compliant = (
            cache_info.has_get_usable_length
            and cache_info.has_get_seq_length
        )

        report.cache_contract = cache_info

        if not cache_info.is_compliant:
            missing = []
            if not cache_info.has_get_usable_length:
                missing.append("get_usable_length")
            if not cache_info.has_get_seq_length:
                missing.append("get_seq_length")

            if self.policy.allow_cache():
                report.add_warning(
                    f"Cache contract not fully satisfied, missing: {missing}. "
                    f"KV caching may fail at runtime."
                )
            else:
                # Not an issue since we're not using cache anyway
                logger.debug(f"Cache contract missing {missing} but cache disabled")

    def _check_config_sync(self, report: CompatibilityReport) -> None:
        """Check that model config is synced with adapter config."""
        if not hasattr(self.model, "config"):
            report.config_synced = False
            return

        expected_attn = self.config.get("attn_implementation", "eager")
        actual_attn = getattr(self.model.config, "attn_implementation", None)

        if actual_attn == expected_attn:
            report.config_synced = True
        else:
            report.config_synced = False
            report.add_warning(
                f"Config attn_implementation mismatch: "
                f"expected '{expected_attn}', got '{actual_attn}'"
            )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt.

        Parameters
        ----------
        prompt : str
            Input prompt.
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature.
        stop : List[str], optional
            Stop sequences.
        **kwargs
            Additional kwargs passed to model.generate().

        Returns
        -------
        str
            Generated text (excluding prompt).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Build generate kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }

        # Apply policy-based cache setting
        if self._force_no_cache:
            gen_kwargs["use_cache"] = False

        # Generate with retry on cache errors
        with torch.no_grad():
            try:
                outputs = self.model.generate(**gen_kwargs)
            except (AttributeError, RuntimeError) as exc:
                msg = str(exc).lower()
                cache_keywords = (
                    "seen_tokens",
                    "get_max_length",
                    "get_usable_length",
                    "dynamiccache",
                    "shape",
                    "size",
                    "mismatch",
                )
                if any(k in msg for k in cache_keywords) and not self._force_no_cache:
                    logger.warning(
                        f"Generation failed ({exc}), retrying with use_cache=False"
                    )
                    self._force_no_cache = True
                    gen_kwargs["use_cache"] = False
                    outputs = self.model.generate(**gen_kwargs)
                else:
                    raise

        # Decode (skip input prompt)
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in generated:
                    generated = generated[: generated.index(stop_seq)]

        return generated.strip()

    def unload(self) -> None:
        """Unload model to free memory."""
        if hasattr(self, "model"):
            del self.model
            self.model = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

        # Try to free GPU memory
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def is_verified(self) -> bool:
        """Return True if model passed all compatibility checks."""
        if self._compatibility_report is None:
            return False
        return self._compatibility_report.is_compatible


__all__ = [
    "ModelAdapter",
    "InvariantViolation",
    "CompatibilityReport",
    "AttentionInfo",
    "CacheContractInfo",
]
