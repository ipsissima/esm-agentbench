#!/usr/bin/env python3
"""Local inference backend for Hugging Face models.

Supports both transformers and vLLM backends with automatic fallback.
All operations are local - no external API calls.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Compatibility patch for DynamicCache.seen_tokens
# In transformers >= 4.41, seen_tokens was removed from DynamicCache.
# Some model code (e.g., Phi-3 with trust_remote_code=True) still uses it.
# Add it back as a property that calls get_seq_length() for backwards compatibility.
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        logger.debug("Patched DynamicCache.seen_tokens for backwards compatibility")
except ImportError:
    pass  # transformers not installed yet


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

    def load(self):
        """Load model using transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: "
                "pip install transformers accelerate"
            )

        logger.info(f"Loading {self.config.hf_id} with transformers backend...")

        # Check if we should use device_map="auto"
        # Only useful with CUDA and requires accelerate to work properly
        use_device_map = False
        if torch.cuda.is_available():
            try:
                from accelerate import infer_auto_device_map
                use_device_map = True
            except ImportError:
                logger.warning("accelerate not fully available, skipping device_map")
        else:
            logger.info("Running on CPU, skipping device_map")

        # Prepare kwargs
        model_kwargs = {
            "trust_remote_code": True,
        }

        # Only use device_map if CUDA available and accelerate works
        if use_device_map:
            model_kwargs["device_map"] = "auto"

        # Handle dtype
        if self.config.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        # Handle quantization
        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("Using 4-bit quantization")
            except ImportError:
                warnings.warn("bitsandbytes not available, skipping quantization")
        elif self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            logger.info("Using 8-bit quantization")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_id,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_id,
            **model_kwargs,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model {self.config.name} loaded successfully")

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, stop: Optional[List[str]] = None) -> str:
        """Generate text from prompt using transformers."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

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

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

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
