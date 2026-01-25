#!/usr/bin/env python3
"""Persistent model worker for efficient generation.

This worker loads the model ONCE at startup and serves multiple generation
requests via Unix domain socket. This eliminates the per-request model reload
overhead (~25-30 seconds per call).

The master process (agent_loop) connects, sends a request, and waits with
timeout. If the worker hangs, the master kills and restarts it.

Usage:
    # Start worker (background)
    python model_worker.py &

    # Or with explicit settings
    ESM_WORKER_SOCKET=/tmp/my_worker.sock \
    ESM_MODEL_ID=microsoft/Phi-3-mini-128k-instruct \
    python model_worker.py

Environment variables:
    ESM_WORKER_SOCKET: Unix socket path (default: /tmp/esm_model_worker.sock)
    ESM_MODEL_ID: HuggingFace model ID (default: microsoft/Phi-3-mini-128k-instruct)
    ESM_MODEL_REVISION: Model revision/commit SHA (optional)
    ESM_WORKER_MAX_TOKENS: Default max tokens (default: 128)
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import time
import traceback
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
SOCKET_PATH = os.environ.get("ESM_WORKER_SOCKET", "/tmp/esm_model_worker.sock")
MODEL_ID = os.environ.get("ESM_MODEL_ID", "microsoft/Phi-3-mini-128k-instruct")
MODEL_REVISION = os.environ.get("ESM_MODEL_REVISION", None)
DEFAULT_MAX_TOKENS = int(os.environ.get("ESM_WORKER_MAX_TOKENS", "128"))

# Global model and tokenizer (loaded once)
model = None
tokenizer = None


def load_model():
    """Load model and tokenizer once at startup."""
    global model, tokenizer

    logger.info(f"Loading model: {MODEL_ID}")
    if MODEL_REVISION:
        logger.info(f"Using revision: {MODEL_REVISION}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()

    # Load tokenizer
    tokenizer_kwargs = {"trust_remote_code": True}
    if MODEL_REVISION:
        tokenizer_kwargs["revision"] = MODEL_REVISION

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - CPU optimized settings
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32,  # float32 for CPU stability
        "low_cpu_mem_usage": True,
    }
    if MODEL_REVISION:
        model_kwargs["revision"] = MODEL_REVISION

    # CPU-specific settings
    if not torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "eager"  # No flash-attn on CPU
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s")

    # Log memory usage
    try:
        import psutil
        proc = psutil.Process()
        mem_gb = proc.memory_info().rss / (1024 ** 3)
        logger.info(f"Worker memory usage: {mem_gb:.2f} GB")
    except ImportError:
        pass


def generate(prompt: str, max_new_tokens: int, temperature: float = 0.7,
             stop_sequences: Optional[list] = None) -> str:
    """Generate text using the loaded model."""
    import torch

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            use_cache=False,  # Safer on CPU
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode (skip input tokens)
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )

    # Handle stop sequences
    if stop_sequences:
        for stop_seq in stop_sequences:
            if stop_seq in generated:
                generated = generated[:generated.index(stop_seq)]

    return generated.strip()


def handle_request(conn: socket.socket) -> None:
    """Handle a single generation request."""
    try:
        # Receive request (JSON terminated by newline)
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break

        if not data:
            logger.warning("Empty request received")
            return

        # Parse request
        request = json.loads(data.decode().strip())
        prompt = request["prompt"]
        max_new_tokens = request.get("max_new_tokens", DEFAULT_MAX_TOKENS)
        temperature = request.get("temperature", 0.7)
        stop_sequences = request.get("stop_sequences", None)

        logger.debug(f"Generating: max_tokens={max_new_tokens}, temp={temperature}")
        t0 = time.time()

        # Generate
        output = generate(prompt, max_new_tokens, temperature, stop_sequences)

        elapsed = time.time() - t0
        logger.info(f"Generated {len(output)} chars in {elapsed:.1f}s")

        # Send response
        response = json.dumps({"ok": True, "output": output})
        conn.sendall((response + "\n").encode())

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON request: {e}")
        error_response = json.dumps({"ok": False, "error": f"Invalid JSON: {e}"})
        conn.sendall((error_response + "\n").encode())

    except Exception as e:
        logger.error(f"Generation error: {e}")
        logger.error(traceback.format_exc())
        error_response = json.dumps({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        conn.sendall((error_response + "\n").encode())

    finally:
        conn.close()


def cleanup(signum=None, frame=None):
    """Clean up socket on exit."""
    logger.info("Shutting down worker...")
    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
            logger.info(f"Removed socket: {SOCKET_PATH}")
        except Exception as e:
            logger.warning(f"Failed to remove socket: {e}")
    sys.exit(0)


def main():
    """Main worker loop."""
    # Register cleanup handlers
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Load model first
    load_model()

    # Remove existing socket if present
    if os.path.exists(SOCKET_PATH):
        logger.warning(f"Removing existing socket: {SOCKET_PATH}")
        os.unlink(SOCKET_PATH)

    # Create Unix domain socket server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o660)
    server.listen(1)

    logger.info(f"Worker listening on: {SOCKET_PATH}")
    logger.info("Ready to serve generation requests")

    try:
        while True:
            conn, _ = server.accept()
            logger.debug("Connection accepted")
            handle_request(conn)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        server.close()
        cleanup()


if __name__ == "__main__":
    main()
