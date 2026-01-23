#!/usr/bin/env python3
"""Subprocess worker script for isolated model generation.

This script loads a model and generates text in a completely isolated subprocess.
Called by agent_loop.py via subprocess.run() to avoid multiprocessing.Queue
semaphores and resource_tracker warnings.

Usage:
    python generate_worker.py \
        --model_id <hf_model_id> \
        --prompt_file <path_to_prompt.txt> \
        --max_new_tokens 64 \
        [--revision <commit_sha>]

Output:
    JSON on stdout: {"output": "<generated_text>"}
    Errors go to stderr and non-zero exit code.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description="Generate text with HuggingFace model")
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--revision", default=None, help="Model revision/commit SHA")
    parser.add_argument("--prompt_file", required=True, help="Path to file containing prompt")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--stop_sequences", default=None, help="JSON-encoded list of stop sequences")
    args = parser.parse_args()

    # Read prompt from file
    try:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse stop sequences if provided
    stop_sequences = None
    if args.stop_sequences:
        try:
            stop_sequences = json.loads(args.stop_sequences)
        except json.JSONDecodeError as e:
            print(f"Error parsing stop_sequences: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        # Import torch and transformers inside worker for clean interpreter state
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        tokenizer_kwargs = {"trust_remote_code": True}
        if args.revision:
            tokenizer_kwargs["revision"] = args.revision

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            **tokenizer_kwargs,
        )

        # Load model - use float32 on CPU for safety
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
        if args.revision:
            model_kwargs["revision"] = args.revision

        # Check if CUDA available
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        else:
            # CPU - force eager attention to avoid flash-attn requirements
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_kwargs,
        )
        model.eval()

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                use_cache=False,  # Safer on CPU, avoids cache-internal issues
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

        # Output JSON to stdout
        print(json.dumps({"output": generated.strip()}))

    except Exception as e:
        print(f"Generation error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
