#!/usr/bin/env python3
"""Debug script for diagnosing Phi-3 (and other HF models) tensor shape mismatches.

This script helps diagnose tensor shape mismatches by:
1. Loading the model with CPU-safe settings
2. Registering forward hooks on attention modules to log shapes
3. Running a test generation to observe tensor shapes

Usage:
    python tools/debug_phi3_shapes.py [--model MODEL_NAME] [--use-cache]

Examples:
    # Test phi-3-mini-instruct with default settings (no cache)
    python tools/debug_phi3_shapes.py

    # Test with use_cache=True to check cache-related shape issues
    python tools/debug_phi3_shapes.py --use-cache

    # Test a different model
    python tools/debug_phi3_shapes.py --model tiny-test
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phi3-debug")


def register_shape_hooks(model):
    """Register forward hooks on attention modules to log tensor shapes."""
    import torch

    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            try:
                if isinstance(out, torch.Tensor):
                    logger.info(f"[HOOK] {name} out.shape={tuple(out.shape)} dtype={out.dtype}")
                elif isinstance(out, (list, tuple)):
                    logger.info(f"[HOOK] {name} out types: {[type(o).__name__ for o in out]}")
                    for i, o in enumerate(out):
                        if isinstance(o, torch.Tensor):
                            logger.info(
                                f"[HOOK] {name} out[{i}].shape={tuple(o.shape)} dtype={o.dtype}"
                            )
            except Exception as e:
                logger.warning(f"[HOOK] {name} failed to inspect: {e}")

        return hook

    # Attach hooks to attention-related modules
    for n, m in model.named_modules():
        ln = n.lower()
        if any(kw in ln for kw in ("attn", "attention", "self_attn", "flash")):
            hooks.append((n, m.register_forward_hook(make_hook(n))))
            logger.debug(f"Registered hook on: {n}")

    logger.info(f"Registered {len(hooks)} shape hooks on attention modules")
    return hooks


def main():
    parser = argparse.ArgumentParser(description="Debug model tensor shapes")
    parser.add_argument(
        "--model",
        default="phi-3-mini-instruct",
        help="Model name from models.yaml (default: phi-3-mini-instruct)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable use_cache during generation (default: False for safer debugging)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        sys.exit(1)

    # Import our inference module
    try:
        from tools.real_agents_hf.inference import create_backend
    except ImportError:
        # Try relative import if run from tools directory
        sys.path.insert(0, str(__file__).rsplit("/tools/", 1)[0])
        from tools.real_agents_hf.inference import create_backend

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Check for optional packages
    try:
        import transformers

        logger.info(f"transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("transformers not installed")
        sys.exit(1)

    try:
        import bitsandbytes

        logger.info(f"bitsandbytes available: {bitsandbytes.__version__}")
    except ImportError:
        logger.info("bitsandbytes not available")

    try:
        import flash_attn

        logger.info(f"flash_attn available: {flash_attn.__version__}")
    except ImportError:
        logger.info("flash_attn not available")

    # Load the model
    logger.info(f"Loading model: {args.model}")
    try:
        backend = create_backend(args.model)
        backend.load()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(2)

    # Log model config info
    if hasattr(backend.model, "config"):
        cfg = backend.model.config
        logger.info(f"Model config:")
        logger.info(f"  - attn_implementation: {getattr(cfg, 'attn_implementation', 'N/A')}")
        logger.info(f"  - _attn_implementation: {getattr(cfg, '_attn_implementation', 'N/A')}")
        logger.info(f"  - torch_dtype: {getattr(cfg, 'torch_dtype', 'N/A')}")

    # Register hooks
    hooks = register_shape_hooks(backend.model)

    # Run test generation
    prompt = "Test shape debug. " * 5
    logger.info(f"Running test generation (use_cache={args.use_cache})")
    logger.info(f"Prompt: {prompt[:50]}...")

    try:
        with torch.no_grad():
            inputs = backend.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(backend.model.device) for k, v in inputs.items()}

            gen_kwargs = {
                **inputs,
                "max_new_tokens": 8,
                "use_cache": args.use_cache,
                "do_sample": False,
            }

            logger.info("Starting generation...")
            outputs = backend.model.generate(**gen_kwargs)
            logger.info("Generation completed successfully!")

            # Decode output
            generated = backend.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            logger.info(f"Generated text: {generated}")

    except RuntimeError as e:
        error_msg = str(e)
        if "shape" in error_msg.lower() or "size" in error_msg.lower():
            logger.error(f"TENSOR SHAPE MISMATCH DETECTED: {e}")
            logger.error("Check the hook outputs above to identify the failing module")
        else:
            logger.error(f"Runtime error during generation: {e}")
        traceback.print_exc()
        sys.exit(3)
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        traceback.print_exc()
        sys.exit(4)
    finally:
        # Clean up hooks
        for name, h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    logger.info("Debug run completed successfully - no shape mismatches detected")
    sys.exit(0)


if __name__ == "__main__":
    main()
