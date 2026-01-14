#!/usr/bin/env python3
"""Feature utilities for drift detection tuning and evaluation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import trim_mean

from certificates.make_certificate import compute_certificate
from certificates.spectral_prover import compute_detection_statistics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_ALIASES = {
    "gold": "coherent",
    "good": "coherent",
}

GOOD_LABELS = {"coherent", "creative", "gold", "good"}
BAD_LABELS = {"drift", "poison", "starvation", "bad"}


@dataclass
class NormalizationConfig:
    l2_normalize_steps: bool = False
    zscore_per_trace: bool = False
    length_normalize: bool = False
    trim_proportion: float = 0.0


@dataclass
class FeatureRow:
    run_id: str
    scenario: str
    label: str
    valid: bool
    features: Dict[str, float]


def _safe_array(embeddings: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(embeddings), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_trace_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data
    return {"trace": data}


def extract_embeddings(trace_data: Dict[str, Any]) -> Optional[np.ndarray]:
    embeddings = trace_data.get("embeddings")
    if embeddings is None:
        return None
    return _safe_array(embeddings)


def l2_normalize_steps(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def zscore_per_trace(embeddings: np.ndarray) -> np.ndarray:
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (embeddings - mean) / std


def length_normalize(embeddings: np.ndarray) -> np.ndarray:
    length = max(embeddings.shape[0], 1)
    return embeddings / np.sqrt(length)


def apply_normalization(
    embeddings: np.ndarray,
    config: Optional[NormalizationConfig] = None,
) -> np.ndarray:
    if config is None:
        return embeddings

    normalized = embeddings.copy()
    if config.l2_normalize_steps:
        normalized = l2_normalize_steps(normalized)
    if config.zscore_per_trace:
        normalized = zscore_per_trace(normalized)
    if config.length_normalize:
        normalized = length_normalize(normalized)
    return normalized


def compute_embed_norm(
    embeddings: np.ndarray,
    trim_proportion: float = 0.0,
) -> float:
    norms = np.linalg.norm(embeddings, axis=1)
    if trim_proportion > 0:
        return float(trim_mean(norms, proportiontocut=trim_proportion))
    return float(np.mean(norms))


def compute_semantic_drift(
    embeddings: np.ndarray,
    trim_proportion: float = 0.0,
) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    distances = []
    for idx in range(1, embeddings.shape[0]):
        prev = embeddings[idx - 1]
        cur = embeddings[idx]
        if np.linalg.norm(prev) < 1e-12 or np.linalg.norm(cur) < 1e-12:
            distances.append(0.0)
        else:
            distances.append(float(cosine(prev, cur)))
    if not distances:
        return 0.0
    if trim_proportion > 0:
        return float(trim_mean(distances, proportiontocut=trim_proportion))
    return float(np.mean(distances))


def infer_label(label: Optional[str]) -> str:
    if not label:
        return "unknown"
    label_lower = label.lower()
    return LABEL_ALIASES.get(label_lower, label_lower)


def infer_scenario_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "scenarios" in parts:
        idx = parts.index("scenarios")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "submissions" in parts:
        idx = parts.index("submissions")
        if idx + 2 < len(parts):
            return parts[idx + 2]
    return "unknown"


def iter_trace_files(trace_dir: Path) -> Iterable[Tuple[Path, str]]:
    if not trace_dir.exists():
        return
    for label_dir in sorted(trace_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = infer_label(label_dir.name)
        for trace_file in sorted(label_dir.glob("*.json")):
            yield trace_file, label


def discover_trace_dirs(base_dirs: Sequence[Path]) -> List[Path]:
    trace_dirs: List[Path] = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        if base_dir.name.startswith("experiment_traces"):
            trace_dirs.append(base_dir)
        for subdir in base_dir.glob("**/experiment_traces*"):
            if subdir.is_dir():
                trace_dirs.append(subdir)
    return sorted(set(trace_dirs))


def compute_trace_features(
    embeddings: np.ndarray,
    k: int = 10,
    normalization: Optional[NormalizationConfig] = None,
    kernel_strict: bool = False,
) -> Dict[str, float]:
    processed = apply_normalization(embeddings, normalization)
    stats = compute_detection_statistics(processed, k=k)
    cert = compute_certificate(processed, r=k, kernel_strict=kernel_strict)

    trim_prop = normalization.trim_proportion if normalization else 0.0

    # core scalar values (safe casts)
    theoretical_bound = float(stats.get("theoretical_bound", np.nan))
    residual = float(stats.get("residual", np.nan))
    pca_explained = float(stats.get("pca_explained", np.nan))
    tail_energy = float(stats.get("tail_energy", np.nan))
    sigma_max = float(stats.get("sigma_max", np.nan))
    singular_gap = float(stats.get("singular_gap", np.nan))
    length_T = float(stats.get("length_T", processed.shape[0]))

    # certificate r_eff may be NaN
    r_eff_value = cert.get("r_eff", np.nan)
    try:
        r_eff = float(r_eff_value) if r_eff_value is not None else np.nan
    except Exception:
        r_eff = np.nan

    # Derived normalizations (robust to zeros/NaN)
    T = max(int(processed.shape[0]), 1)
    sqrt_T = float(np.sqrt(T))

    # residual normalized by sqrt(T)
    if np.isnan(residual):
        residual_norm = float(np.nan)
    else:
        residual_norm = float(residual / sqrt_T)

    # residual normalized by Frobenius norm of X
    fro = float(np.linalg.norm(processed))
    if fro < 1e-12:
        residual_fro_norm = float(np.nan)
    else:
        residual_fro_norm = float(residual / fro) if not np.isnan(residual) else float(np.nan)

    # theoretical bound normalized by sqrt(r_eff) (fallback safe)
    if np.isnan(theoretical_bound) or (np.isnan(r_eff) or r_eff <= 0.0):
        theoretical_bound_norm = float(np.nan)
    else:
        theoretical_bound_norm = float(theoretical_bound / np.sqrt(max(1.0, r_eff)))

    # effective rank relative to length
    r_rel = float(np.nan)
    if not np.isnan(r_eff) and T > 0:
        r_rel = float(r_eff / float(T))

    # largest singular-value ratio for poison detection
    sv_max_ratio = float(np.nan)
    try:
        svals = np.linalg.svd(processed, compute_uv=False)
        total = float(np.sum(svals**2))
        if total > 0:
            sv_max_ratio = float((svals[0]**2) / total)
    except Exception:
        sv_max_ratio = float(np.nan)

    return {
        "theoretical_bound": theoretical_bound,
        "residual": residual,
        "koopman_residual": float(stats.get("koopman_residual", np.nan)),
        "pca_explained": pca_explained,
        "tail_energy": tail_energy,
        "sigma_max": sigma_max,
        "singular_gap": singular_gap,
        "length_T": length_T,
        "r_eff": r_eff,
        "r_rel": r_rel,
        "insample_residual": float(cert.get("insample_residual", np.nan)),
        "oos_residual": float(cert.get("oos_residual", np.nan)),
        "embed_norm": compute_embed_norm(processed, trim_proportion=trim_prop),
        "semantic_drift": compute_semantic_drift(processed, trim_proportion=trim_prop),
        # NEW normalized features
        "residual_norm": residual_norm,
        "residual_fro_norm": residual_fro_norm,
        "theoretical_bound_norm": theoretical_bound_norm,
        "sv_max_ratio": sv_max_ratio,
    }


def label_to_binary(label: str) -> Optional[int]:
    label_norm = infer_label(label)
    if label_norm in GOOD_LABELS:
        return 0
    if label_norm in BAD_LABELS:
        return 1
    return None
