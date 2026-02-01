"""HuggingFace inference adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ports.inference import InferencePort


class HuggingFaceInferenceAdapter(InferencePort):
    """Adapter for HuggingFace model inference.
    
    This adapter wraps the HuggingFace transformers backend to implement
    the InferencePort interface for the hexagonal architecture.
    """

    def __init__(self, model_name: str, config_path: Optional[Path] = None) -> None:
        """Initialize the HuggingFace inference adapter.
        
        Parameters
        ----------
        model_name : str
            Name of the model to use (from models.yaml config).
        config_path : Path, optional
            Path to models configuration file.
        """
        self._model_name = model_name
        self._config_path = config_path
        self._backend = None

    def _ensure_backend(self) -> None:
        """Lazy-load the inference backend."""
        if self._backend is None:
            try:
                # Import here to avoid circular dependencies
                import sys
                from pathlib import Path
                
                # Add tools/real_agents_hf to path if not already there
                tools_path = Path(__file__).parent.parent / "tools" / "real_agents_hf"
                if str(tools_path) not in sys.path:
                    sys.path.insert(0, str(tools_path))
                
                from inference import create_backend
                
                self._backend = create_backend(self._model_name, self._config_path)
                self._backend.load()
            except ImportError as exc:
                raise RuntimeError(
                    f"Failed to load HuggingFace backend: {exc}. "
                    "Ensure transformers and related dependencies are installed."
                ) from exc

    def run_inference(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Run inference and return a structured response.
        
        Parameters
        ----------
        prompt : str
            The input prompt for inference.
        system_prompt : str, optional
            System prompt to prepend (if supported by model).
        parameters : Mapping[str, Any], optional
            Additional inference parameters (e.g., temperature, max_tokens).
            
        Returns
        -------
        Mapping[str, Any]
            Structured response containing:
            - "text": Generated text
            - "model": Model name used
            - "parameters": Parameters used for generation
        """
        self._ensure_backend()
        
        # Merge system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Extract parameters or use defaults
        params = parameters or {}
        max_tokens = params.get("max_tokens")
        temperature = params.get("temperature")
        stop = params.get("stop")
        
        # Generate response
        generated_text = self._backend.generate(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        
        # Return structured response
        return {
            "text": generated_text,
            "model": self._model_name,
            "parameters": {
                "max_tokens": max_tokens or self._backend.config.max_tokens,
                "temperature": temperature or self._backend.config.temperature,
                "stop": stop,
            },
        }

    def unload(self) -> None:
        """Unload the model to free resources."""
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
