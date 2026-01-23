#!/usr/bin/env python3
"""Debug script to measure single generate() call timing.

Use this to isolate whether the generate() call itself is slow or
if the slowness comes from the agent loop overhead.

Usage:
    python tools/debug_generate.py [--model phi-3-mini-instruct]
    python tools/debug_generate.py --model phi-3-mini-instruct --with-timeout
    python tools/debug_generate.py --system-info-only  # Just show system resources
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def log_system_info():
    """Log detailed system information for diagnostics."""
    logger.info("-" * 40)
    logger.info("SYSTEM INFORMATION")
    logger.info("-" * 40)

    # Memory info
    try:
        with open('/proc/meminfo') as f:
            meminfo = {}
            for line in f:
                if ':' in line:
                    key, val = line.split(':', 1)
                    meminfo[key.strip()] = val.strip()

        total_kb = int(meminfo.get('MemTotal', '0').split()[0])
        avail_kb = int(meminfo.get('MemAvailable', '0').split()[0])
        free_kb = int(meminfo.get('MemFree', '0').split()[0])
        buffers_kb = int(meminfo.get('Buffers', '0').split()[0])
        cached_kb = int(meminfo.get('Cached', '0').split()[0])
        swap_total_kb = int(meminfo.get('SwapTotal', '0').split()[0])
        swap_free_kb = int(meminfo.get('SwapFree', '0').split()[0])

        logger.info(f"  Memory Total:     {total_kb / (1024**2):.1f} GB")
        logger.info(f"  Memory Available: {avail_kb / (1024**2):.1f} GB")
        logger.info(f"  Memory Free:      {free_kb / (1024**2):.1f} GB")
        logger.info(f"  Buffers/Cached:   {(buffers_kb + cached_kb) / (1024**2):.1f} GB")
        logger.info(f"  Swap Total:       {swap_total_kb / (1024**2):.1f} GB")
        logger.info(f"  Swap Used:        {(swap_total_kb - swap_free_kb) / (1024**2):.1f} GB")

        # Memory pressure indicator
        used_pct = ((total_kb - avail_kb) / total_kb) * 100
        if used_pct > 90:
            logger.warning(f"  Memory pressure: CRITICAL ({used_pct:.1f}% used)")
        elif used_pct > 75:
            logger.warning(f"  Memory pressure: HIGH ({used_pct:.1f}% used)")
        else:
            logger.info(f"  Memory pressure: OK ({used_pct:.1f}% used)")
    except Exception as e:
        logger.warning(f"  Could not read /proc/meminfo: {e}")

    # CPU info
    try:
        cpu_count = os.cpu_count() or 1
        logger.info(f"  CPU cores: {cpu_count}")

        # Load average
        with open('/proc/loadavg') as f:
            loadavg = f.read().split()[:3]
            logger.info(f"  Load average: {' '.join(loadavg)}")
            load_1m = float(loadavg[0])
            if load_1m > cpu_count * 2:
                logger.warning(f"  System load: HIGH (load {load_1m} > {cpu_count * 2})")
    except Exception as e:
        logger.warning(f"  Could not read CPU info: {e}")

    # PyTorch/CUDA info
    try:
        import torch
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            mem_alloc = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"  GPU memory allocated: {mem_alloc:.2f} GB")
            logger.info(f"  GPU memory reserved: {mem_reserved:.2f} GB")
    except ImportError:
        logger.info("  PyTorch: not installed")
    except Exception as e:
        logger.warning(f"  Could not get PyTorch info: {e}")

    logger.info("-" * 40)


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
    parser.add_argument(
        "--with-timeout",
        action="store_true",
        help="Use the timeout wrapper around generate()",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds when using --with-timeout (default: 300)",
    )
    parser.add_argument(
        "--system-info-only",
        action="store_true",
        help="Only show system info, don't run generate",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Debug Generate Timing Script")
    logger.info("=" * 60)

    # Always show system info first
    log_system_info()

    if args.system_info_only:
        logger.info("System info only mode - exiting")
        return

    # Import after argparse to catch import errors separately
    try:
        from tools.real_agents_hf.inference import create_backend
        if args.with_timeout:
            from tools.real_agents_hf.generation_timeout import generate_with_timeout, GenerationTimeout
    except ImportError:
        # Try relative import if running from tools directory
        sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
        from tools.real_agents_hf.inference import create_backend
        if args.with_timeout:
            from tools.real_agents_hf.generation_timeout import generate_with_timeout, GenerationTimeout

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
    if args.with_timeout:
        logger.info(f"  Using timeout wrapper: {args.timeout}s")

    t0 = time.time()
    result = None
    t_generate = 0.0

    try:
        if args.with_timeout:
            result = generate_with_timeout(
                backend, prompt, max_tokens=args.max_tokens,
                timeout_seconds=args.timeout
            )
        else:
            result = backend.generate(prompt, max_tokens=args.max_tokens)
        t_generate = time.time() - t0
        logger.info(f"Generate completed in {t_generate:.2f}s")
        logger.info(f"  Output length: {len(result)} chars")
        logger.info(f"  Output preview: {result[:200]!r}...")
    except Exception as e:
        t_generate = time.time() - t0
        error_type = type(e).__name__
        if error_type == "GenerationTimeout":
            logger.error(f"TIMEOUT: generate() did not complete in {args.timeout}s")
            logger.error(f"  Elapsed before timeout: {t_generate:.2f}s")
            # Log system state at timeout
            log_system_info()
            backend.unload()
            sys.exit(2)
        else:
            logger.error(f"ERROR: generate() failed after {t_generate:.2f}s: {e}")
            backend.unload()
            sys.exit(3)

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
