#!/usr/bin/env python3
"""Local embedding generation for agent traces.

Uses local Hugging Face embedding models (no API calls).
Implements caching to avoid recomputing embeddings.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Local embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: Optional[Path] = None):
        """Initialize embedding model.

        Parameters
        ----------
        model_name : str
            Hugging Face model ID for embeddings
        cache_dir : Path, optional
            Directory to cache embeddings
        """
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: "
                "pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded")

    def _get_cache_path(self, text: str) -> Optional[Path]:
        """Get cache file path for text."""
        if not self.cache_dir:
            return None

        # Hash text to get cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.npy"

    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache if available."""
        cache_path = self._get_cache_path(text)
        if cache_path and cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        return None

    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_path = self._get_cache_path(text)
        if cache_path:
            try:
                np.save(cache_path, embedding)
            except Exception as e:
                logger.warning(f"Failed to save to cache: {e}")

    def embed(self, text: str) -> np.ndarray:
        """Compute embedding for text.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        np.ndarray
            Embedding vector
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Try cache first
        cached = self._load_from_cache(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Cache it
        self._save_to_cache(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Compute embeddings for multiple texts.

        Parameters
        ----------
        texts : list of str
            Texts to embed
        batch_size : int
            Batch size for encoding

        Returns
        -------
        list of np.ndarray
            Embedding vectors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embeddings = []
        to_compute = []
        to_compute_indices = []

        # Check cache
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                to_compute.append(text)
                to_compute_indices.append(i)

        # Compute missing embeddings
        if to_compute:
            logger.info(f"Computing {len(to_compute)} embeddings (batch_size={batch_size})")
            computed = self.model.encode(
                to_compute,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=len(to_compute) > 100,
            )

            # Cache and add to results
            for text, emb, idx in zip(to_compute, computed, to_compute_indices):
                self._save_to_cache(text, emb)
                embeddings.append((idx, emb))

        # Sort by original index and return
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]

    def get_embedder_id(self) -> str:
        """Get canonical embedder ID for provenance tracking.
        
        Computes embedder_id as "{model_name}|{model_files_sha256}" where
        model_files_sha256 is the SHA256 over the concatenation of file SHA256s
        in the local model folder. Falls back to "unknown" if files are not cached.
        
        Returns
        -------
        str
            Embedder ID in format "model_name|hash" or "model_name|unknown"
        """
        model_files_sha = "unknown"
        
        # Try to compute hash of local model files if available
        if self.model is not None:
            try:
                # Get the model cache directory from sentence-transformers
                # The model stores its path in various places depending on version
                model_cache_path = None
                
                # Try multiple approaches to find the cache directory
                if hasattr(self.model, '_first_module'):
                    first_module = self.model._first_module()
                    if hasattr(first_module, 'auto_model'):
                        if hasattr(first_module.auto_model, 'config'):
                            config = first_module.auto_model.config
                            if hasattr(config, '_name_or_path'):
                                model_cache_path = Path(config._name_or_path)
                
                # Fallback: check if cache_dir was explicitly provided
                if model_cache_path is None and self.cache_dir is not None:
                    # This is embedding cache, not model cache, but try anyway
                    potential_model_cache = self.cache_dir.parent / "models" / self.model_name.replace("/", "--")
                    if potential_model_cache.exists():
                        model_cache_path = potential_model_cache
                
                # Compute hash if we found a valid cache directory
                if model_cache_path and model_cache_path.exists() and model_cache_path.is_dir():
                    # Collect all files in sorted order
                    all_files = sorted(model_cache_path.rglob("*"))
                    file_hashes = []
                    
                    for file_path in all_files:
                        if file_path.is_file():
                            # Compute SHA256 of file content
                            file_hash = hashlib.sha256()
                            with open(file_path, 'rb') as f:
                                # Read in chunks to handle large files
                                while chunk := f.read(8192):
                                    file_hash.update(chunk)
                            file_hashes.append(file_hash.digest())
                    
                    if file_hashes:
                        # Concatenate all file hashes and compute final SHA256
                        combined_hash = hashlib.sha256(b''.join(file_hashes))
                        model_files_sha = combined_hash.hexdigest()
                        
            except Exception as e:
                logger.warning(f"Could not compute model file hash: {e}")
                model_files_sha = "unknown"
        
        return f"{self.model_name}|{model_files_sha}"

    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def embed_trace_steps(
    steps: List[dict],
    embedding_model: EmbeddingModel,
) -> List[np.ndarray]:
    """Compute embeddings for trace steps.

    Parameters
    ----------
    steps : list of dict
        Agent steps from trace
    embedding_model : EmbeddingModel
        Loaded embedding model

    Returns
    -------
    list of np.ndarray
        Embeddings for each step
    """
    # Extract text from each step
    texts = []
    for step in steps:
        # Combine type and text for richer embedding
        step_type = step.get('type', 'unknown')
        text = step.get('text', '')

        # Add tool info if present
        if 'tool' in step:
            tool = step['tool']
            args = step.get('args', {})
            text = f"{step_type}: {tool}({args}) - {text}"
        else:
            text = f"{step_type}: {text}"

        texts.append(text)

    # Compute embeddings
    return embedding_model.embed_batch(texts)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test embedding model
    model = EmbeddingModel(cache_dir=Path("/tmp/embedding_cache"))
    model.load()

    # Test single embedding
    text = "This is a test sentence for embedding."
    emb = model.embed(text)
    logger.info(f"Embedding shape: {emb.shape}")
    logger.info(f"First 5 values: {emb[:5]}")

    # Test batch embedding
    texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence",
    ]
    embs = model.embed_batch(texts)
    logger.info(f"Batch embeddings: {len(embs)} vectors of shape {embs[0].shape}")

    model.unload()
