# tools/real_agents_hf/shims.py
"""
Compatibility shims for transformers internals.

This module centralizes all monkeypatches needed for compatibility with
different versions of transformers and remote model code. All shims are
applied via apply_transformers_shims() which should be called early
(e.g., at module import time in inference.py).

Shims included:
- DynamicCache.seen_tokens: Property for backwards compatibility
- DynamicCache.get_max_length: Method for remote model code
- DynamicCache.get_usable_length: Method accepting seq_length arg

Design notes:
- Each shim is idempotent (safe to apply multiple times)
- Shims fail silently if transformers is not installed
- All shims are documented with the invariant they preserve
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_shims_applied = False


def _patch_seen_tokens() -> bool:
    """Patch DynamicCache.seen_tokens for transformers >= 4.41 compatibility.

    In transformers >= 4.41, seen_tokens was removed from DynamicCache.
    Some model code (e.g., Phi-3 with trust_remote_code=True) still uses it.

    Invariant: cache.seen_tokens == cache.get_seq_length()
    """
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, 'seen_tokens'):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
            logger.debug("Patched DynamicCache.seen_tokens")
            return True
        return False
    except ImportError:
        logger.debug("DynamicCache.seen_tokens shim unavailable (transformers not installed)")
        return False
    except Exception as e:
        logger.debug("DynamicCache.seen_tokens shim failed: %s", e)
        return False


def _patch_get_max_length() -> bool:
    """Patch DynamicCache.get_max_length for remote model code.

    Some remote modeling code expects get_max_length() method.
    Provides safe fallbacks using available methods.

    Invariant: Returns a non-negative integer representing cache capacity.
    """
    try:
        from transformers.cache_utils import DynamicCache
        if hasattr(DynamicCache, 'get_max_length'):
            return False

        def _get_max_length(self: Any) -> int:
            # Prefer explicit capacity method if present
            if hasattr(self, 'get_capacity'):
                try:
                    return int(self.get_capacity())
                except Exception:
                    pass
            # Try max_length attribute
            if hasattr(self, 'max_length'):
                try:
                    return int(self.max_length)
                except Exception:
                    pass
            # Fall back to current sequence length as conservative proxy
            if hasattr(self, 'get_seq_length'):
                try:
                    return int(self.get_seq_length())
                except Exception:
                    pass
            # Last resort: try len(self)
            try:
                return int(len(self))
            except Exception:
                return 0

        DynamicCache.get_max_length = _get_max_length
        logger.debug("Patched DynamicCache.get_max_length")
        return True
    except ImportError:
        logger.debug("DynamicCache.get_max_length shim unavailable")
        return False
    except Exception as e:
        logger.debug("DynamicCache.get_max_length shim failed: %s", e)
        return False


def _patch_get_usable_length() -> bool:
    """Patch DynamicCache.get_usable_length for remote model code.

    Remote model code sometimes calls get_usable_length(seq_length).
    This shim accepts both positional and keyword seq_length arguments.

    Invariant: Returns a non-negative integer <= capacity (if known).
    """
    try:
        from transformers.cache_utils import DynamicCache
        if hasattr(DynamicCache, 'get_usable_length'):
            return False

        def _get_usable_length(self: Any, *args: Any, **kwargs: Any) -> int:
            # Accept either positional or keyword 'seq_length'
            seq_length = None
            if args:
                seq_length = args[0]
            elif "seq_length" in kwargs:
                seq_length = kwargs["seq_length"]

            try:
                if seq_length is not None:
                    try:
                        seq_val = int(seq_length)
                    except Exception:
                        seq_val = None
                    if seq_val is not None:
                        # Clamp to capacity if available
                        if hasattr(self, "get_capacity"):
                            try:
                                cap = int(self.get_capacity())
                                return min(seq_val, cap)
                            except Exception:
                                return seq_val
                        return seq_val

                # No seq_length provided: try conservative fallbacks
                if hasattr(self, "get_seq_length"):
                    return int(self.get_seq_length())
                if hasattr(self, "get_capacity"):
                    return int(self.get_capacity())
                try:
                    return int(len(self))
                except Exception:
                    return 0
            except Exception:
                return 0

        DynamicCache.get_usable_length = _get_usable_length
        logger.debug("Patched DynamicCache.get_usable_length")
        return True
    except ImportError:
        logger.debug("DynamicCache.get_usable_length shim unavailable")
        return False
    except Exception as e:
        logger.debug("DynamicCache.get_usable_length shim failed: %s", e)
        return False


def apply_transformers_shims() -> dict:
    """Apply all transformers compatibility shims.

    This function is idempotent and safe to call multiple times.
    Should be called early, before any model loading.

    Returns
    -------
    dict
        Status of each shim application: {shim_name: was_applied}
    """
    global _shims_applied

    if _shims_applied:
        logger.debug("Transformers shims already applied, skipping")
        return {"already_applied": True}

    results = {
        "seen_tokens": _patch_seen_tokens(),
        "get_max_length": _patch_get_max_length(),
        "get_usable_length": _patch_get_usable_length(),
    }

    applied = [k for k, v in results.items() if v]
    if applied:
        logger.info("Applied transformers shims: %s", ", ".join(applied))
    else:
        logger.debug("No new transformers shims needed")

    _shims_applied = True
    return results


def reset_shims_state() -> None:
    """Reset shims state (for testing only)."""
    global _shims_applied
    _shims_applied = False


__all__ = [
    "apply_transformers_shims",
    "reset_shims_state",
]
