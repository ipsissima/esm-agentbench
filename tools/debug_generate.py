#!/usr/bin/env python3
"""Debug script to measure single generate() call timing.

Use this to isolate whether the generate() call itself is slow or
if the slowness comes from the agent loop overhead.

Usage:
    python tools/debug_generate.py [--model phi-3-mini-instruct]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Debug single generate() timing")
    parser.add_argument(
        "--model",
        default="phi-3-mini-instruct",
        help="Model name from models.yaml (default: phi-3-mini-instruct)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate (default: 50)",
    )
    args = parser.parse_args()

    # Import after argparse to catch import errors separately
    try:
        from tools.real_agents_hf.inference import create_backend
    except ImportError:
        # Try relative import if running from tools directory
        sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
        from tools.real_agents_hf.inference import create_backend

    logger.info("=" * 60)
    logger.info("Debug Generate Timing Script")
    logger.info("=" * 60)

    # Create backend
    logger.info(f"Creating backend for model: {args.model}")
    t0 = time.time()
    backend = create_backend(args.model)
    t_create = time.time() - t0
    logger.info(f"Backend created in {t_create:.2f}s")

    # Load model
    logger.info("Loading model...")
    t0 = time.time()
    backend.load()
    t_load = time.time() - t0
    logger.info(f"Model loaded in {t_load:.2f}s")

    # Simple test prompt
    prompt = """You are a coding assistant. Respond briefly.

Task: Write a Python function that adds two numbers.

THOUGHT:"""

    # Run generate() and time it
    logger.info("Running generate()...")
    logger.info(f"  Prompt length: {len(prompt)} chars")
    logger.info(f"  Max tokens: {args.max_tokens}")

    t0 = time.time()
    result = backend.generate(prompt, max_tokens=args.max_tokens)
    t_generate = time.time() - t0

    logger.info(f"Generate completed in {t_generate:.2f}s")
    logger.info(f"  Output length: {len(result)} chars")
    logger.info(f"  Output preview: {result[:200]!r}...")

    # Unload
    logger.info("Unloading model...")
    backend.unload()

    # Summary
    logger.info("=" * 60)
    logger.info("TIMING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Backend creation: {t_create:.2f}s")
    logger.info(f"  Model load:       {t_load:.2f}s")
    logger.info(f"  Generate call:    {t_generate:.2f}s")
    logger.info(f"  TOTAL:            {t_create + t_load + t_generate:.2f}s")
    logger.info("=" * 60)

    # Check if generate is suspiciously slow
    if t_generate > 300:  # 5 minutes
        logger.error(f"CRITICAL: generate() took {t_generate:.0f}s (>5 minutes)")
        logger.error("Consider using a smaller model or reducing workload")
        sys.exit(1)
    elif t_generate > 120:  # 2 minutes
        logger.warning(f"WARNING: generate() took {t_generate:.0f}s (>2 minutes)")

    logger.info("Debug script completed successfully")


if __name__ == "__main__":
    main()
