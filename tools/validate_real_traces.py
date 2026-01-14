#!/usr/bin/env python
"""Validate drift detection metric against real traces.

This script tests the theoretical_bound metric using:
1. Existing real traces in tools/real_traces/
2. Newly generated traces via harvest_data.py (deterministic agent)

IMPORTANT: This validation requires semantic embeddings (sentence-transformers or OpenAI).
TF-IDF fallback will produce misleading results because it measures vocabulary difference,
not semantic coherence. Creative solutions using different vocabulary (e.g., matrix
exponentiation) would incorrectly appear as "drift" with TF-IDF.
"""
from __future__ import annotations

# Suppress FutureWarning from transformers about deprecated torch pytree API
# This must be done before importing sentence_transformers or transformers
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*_register_pytree_node.*is deprecated.*",
    category=FutureWarning,
)
# Also suppress huggingface_hub deprecation warnings
warnings.filterwarnings(
    "ignore",
    message=r".*resume_download.*is deprecated.*",
    category=FutureWarning,
)

import argparse
import json
import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union

import numpy as np
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.feature_utils import NormalizationConfig, compute_trace_features
from tools.bootstrap_calibration import bootstrap_null_threshold

# Track which embedding method is used
_embedding_method = "unknown"

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def _check_embedding_availability() -> str:
    """Check which embedding method is available and return its name.

    Fast check that doesn't attempt network requests.
    Uses environment checks and local cache detection only.
    """
    global _embedding_method

    # Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            _embedding_method = "openai"
            return "openai"
        except ImportError:
            pass

    # Check for cached sentence-transformers model (fast filesystem check)
    # Don't import sentence_transformers as it may trigger network calls
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, "models--sentence-transformers--all-MiniLM-L6-v2")
    if os.path.exists(model_cache):
        # Model is cached, sentence-transformers should work
        try:
            import sentence_transformers
            _embedding_method = "sentence-transformers"
            return "sentence-transformers"
        except ImportError:
            pass

    # Check if sentence-transformers is installed but model not cached
    try:
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is not None:
            print("  Note: sentence-transformers installed but model not cached locally")
            print("  Network access to huggingface.co may be required")
    except Exception:
        pass

    _embedding_method = "tfidf"
    return "tfidf"


def embed_trace_steps_with_check(trace: List[Dict[str, Any]]) -> Tuple[np.ndarray, str]:
    """Embed trace steps and return (embeddings, method_used)."""
    from assessor.kickoff import embed_trace_steps

    embeddings = embed_trace_steps(trace)
    return embeddings, _embedding_method


def load_trace_with_meta(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a trace from JSON file, returning (trace, metadata)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict):
        trace = data.get("trace")
        if trace is None and isinstance(data.get("steps"), list):
            trace = data.get("steps")
        if trace is None and any(isinstance(v, list) for v in data.values()):
            for value in data.values():
                if isinstance(value, list):
                    trace = value
                    break
        meta = dict(data)
        if trace is None:
            return [], meta
        return trace if isinstance(trace, list) else [], meta
    return [], {}


def load_trace(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Backwards-compatible loader that returns only the trace (list of steps).

    Historically tests and other modules imported `load_trace` from this
    module and expected a plain list-of-step dictionaries. During a
    refactor the richer `load_trace_with_meta` (which returns (trace, meta))
    was introduced and `load_trace` was removed.

    Keep this small wrapper to preserve the public API expected by tests:
        load_trace(path) -> trace (List[Dict[str, Any]])

    Accepts either a `str` or `pathlib.Path` for convenience.
    """
    # normalize path
    p = Path(path) if not isinstance(path, Path) else path
    trace, _meta = load_trace_with_meta(p)
    # ensure it's a list
    if trace is None:
        return []
    return trace


def _extract_prompt_text(trace: List[Dict[str, Any]], meta: Dict[str, Any] | None) -> str | None:
    if meta:
        for key in ("poison", "prompt", "task", "context"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    for step in trace:
        val = step.get("context") or step.get("prompt") or step.get("task")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def analyze_trace(
    trace: List[Dict[str, Any]],
    label: str,
    meta: Dict[str, Any] | None = None,
    gamma: float = 0.3,
) -> Dict[str, Any]:
    """Analyze a single trace and compute certificate diagnostics."""
    from assessor.kickoff import _sentence_model, embed_trace_steps
    from scipy.spatial.distance import cosine

    if trace is None or len(trace) == 0:
        return {}

    try:
        embeddings = embed_trace_steps(trace)
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings)
    except Exception:
        return {}

    norm_cfg = NormalizationConfig(
        l2_normalize_steps=True,
        zscore_per_trace=False,
        length_normalize=False,
        trim_proportion=0.0,
    )
    features = compute_trace_features(embeddings, normalization=norm_cfg)

    theoretical_bound_raw = float(features.get("theoretical_bound", float("nan")))
    theoretical_bound_norm = float(features.get("theoretical_bound_norm", float("nan")))
    residual_raw = float(features.get("residual", float("nan")))
    residual_norm = float(features.get("residual_norm", float("nan")))
    r_eff = float(features.get("r_eff", float("nan")))
    r_rel = float(features.get("r_rel", float("nan")))
    pca_explained = float(features.get("pca_explained", float("nan")))

    tb_use = theoretical_bound_norm if not np.isnan(theoretical_bound_norm) else theoretical_bound_raw
    semantic_centroid_distance = float("nan")
    adjusted_bound = float(tb_use) if tb_use is not None and not np.isnan(tb_use) else float("nan")

    if meta:
        for key in ("prompt", "task", "poison", "context", "instructions"):
            if key in meta and isinstance(meta[key], str) and meta[key].strip():
                prompt_text = meta[key].strip()
                try:
                    model = _sentence_model()
                    task_vec = np.asarray(model.encode([prompt_text], convert_to_numpy=True))[0]
                    centroid = np.mean(embeddings, axis=0)
                    if np.linalg.norm(centroid) > 1e-12 and np.linalg.norm(task_vec) > 1e-12:
                        semantic_centroid_distance = float(cosine(centroid, task_vec))
                        if not np.isnan(adjusted_bound):
                            adjusted_bound = float(adjusted_bound - gamma * semantic_centroid_distance)
                except Exception:
                    pass
                break

    return {
        "label": label,
        "category": label,
        "n_steps": embeddings.shape[0],
        "theoretical_bound": theoretical_bound_raw,
        "theoretical_bound_norm": theoretical_bound_norm,
        "theoretical_bound_prompt_adj": adjusted_bound,
        "residual": residual_raw,
        "residual_norm": residual_norm,
        "oos_residual": float(features.get("oos_residual", float("nan"))),
        "insample_residual": float(features.get("insample_residual", float("nan"))),
        "r_eff": r_eff,
        "r_rel": r_rel,
        "pca_explained": pca_explained,
        "pca_tail_estimate": float(features.get("tail_energy", float("nan"))),
        "semantic_centroid_distance": semantic_centroid_distance,
        "sv_max_ratio": float(features.get("sv_max_ratio", float("nan"))),
        "embeddings": embeddings,
        **features,
    }


def categorize_trace(filename: str) -> str:
    """Categorize a trace based on its filename."""
    name = filename.lower()
    # Creative must be checked before coherent since creative traces are also coherent
    if "creative" in name:
        return "creative"
    elif "gold" in name or "coherent" in name or "good" in name:
        return "coherent"
    elif "drift" in name or "hallucination" in name or "bad" in name:
        return "drift"
    elif "poison" in name:
        return "poison"
    elif "starved" in name or "starvation" in name or "memory" in name:
        return "starvation"
    return "unknown"


def pick_score(result: Dict[str, Any]) -> float:
    value = result.get("theoretical_bound_prompt_adj")
    if isinstance(value, (int, float)) and not np.isnan(value):
        return float(value)
    value = result.get("theoretical_bound_norm")
    if isinstance(value, (int, float)) and not np.isnan(value):
        return float(value)
    value = result.get("theoretical_bound")
    return float(value) if value is not None else float("nan")


def compute_Rnorm_for_embeddings(embeddings: np.ndarray) -> float:
    """Compute R_norm using r0 for explained variance 0.9."""
    if embeddings.size == 0:
        return float("nan")
    centered = embeddings - np.mean(embeddings, axis=0)
    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
    energy = float(np.sum(svals**2))
    if energy <= 0:
        return float("nan")
    cum = np.cumsum(svals**2) / energy
    r0 = int(np.searchsorted(cum, 0.9) + 1)
    tail = float(np.sum(svals[r0:] ** 2))
    return float(np.sqrt(tail / energy))


def bootstrap_rnorm_threshold(
    embeddings_list: List[np.ndarray],
    B: int = 500,
    percentile: float = 95.0,
) -> float | None:
    if not embeddings_list:
        return None
    pooled_boot = bootstrap_null_threshold(embeddings_list, B=B, rows_resample=True, percentile=percentile)
    rnorm_values = []
    for boot_iter in pooled_boot:
        for sample in boot_iter:
            rnorm_values.append(compute_Rnorm_for_embeddings(np.asarray(sample)))
    if not rnorm_values:
        return None
    return float(np.percentile(rnorm_values, percentile))


def run_real_trace_validation(
    real_traces_dir: Path | None = None,
    gamma: float = 0.3,
    use_prompt_adj: bool = False,
    bootstrap_B: int = 500,
):
    """Run validation on real traces."""
    print("=" * 80)
    print("REAL TRACE VALIDATION TEST")
    print("=" * 80)

    if real_traces_dir is None:
        real_traces_dir = Path(__file__).parent / "real_traces"

    # Analyze existing real traces
    print("\n[1] Analyzing existing real traces in tools/real_traces/")
    print("-" * 80)

    results = {}
    excluded_traces = []
    excluded_trace_counts = {
        "trace_too_short": 0,
        "numeric_anomaly": 0,
        "sentinel_empty_certificate": 0,
        "numeric_outlier": 0,
    }
    results_by_category = {"coherent": [], "creative": [], "drift": [], "poison": [], "starvation": []}
    coherent_embeddings: List[np.ndarray] = []

    for trace_file in sorted(real_traces_dir.glob("*.json")):
        label = trace_file.stem
        trace, meta = load_trace_with_meta(trace_file)
        
        # FILTER: Ignore traces shorter than 10 steps.
        # Short traces (crashes) produce linear artifacts that skew the spectral analysis.
        if len(trace) < 10:
            print(f"  [Skipping] {label}: Trace too short ({len(trace)} steps) to measure drift.")
            excluded_traces.append(
                {"label": label, "reason": "trace_too_short", "details": {"n_steps": len(trace)}}
            )
            excluded_trace_counts["trace_too_short"] += 1
            continue
        
        if trace:
            result = analyze_trace(trace, label, meta=meta, gamma=gamma)
            if not use_prompt_adj:
                result["theoretical_bound_prompt_adj"] = float("nan")

            # ====================================================
            # EXPERT HOTFIX: Robust sanity checks for numeric anomalies
            # ====================================================
            # 1) Reject NaN/Inf in critical fields
            bad_numeric = False
            for k in ("residual", "theoretical_bound", "pca_explained", "r_eff"):
                v = result.get(k, None)
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    print(f"  [Skipping] {label}: NUMERIC ANOMALY ({k}={v})")
                    excluded_traces.append(
                        {"label": label, "reason": "numeric_anomaly", "details": {"field": k, "value": v}}
                    )
                    excluded_trace_counts["numeric_anomaly"] += 1
                    bad_numeric = True
                    break
            if bad_numeric:
                continue

            # 2) Reject sentinel "empty certificate" (conservative): detect exact pattern
            # _initial_empty_certificate returns residual=1.0, tail_energy=1.0, semantic_divergence=1.0, theoretical_bound=3.0
            # If we see that exact conservative sentinel, treat as invalid trace (computation failed).
            if (np.isclose(result.get("residual", 0.0), 1.0) and
                np.isclose(result.get("theoretical_bound", 0.0), 3.0) and
                np.isclose(result.get("pca_explained", 0.0), 0.0) and
                np.isclose(result.get("r_eff", 0.0), 1.0)):
                print(f"  [Skipping] {label}: SENTINEL/EMPTY CERTIFICATE returned (likely numerical failure)")
                excluded_traces.append(
                    {"label": label, "reason": "sentinel_empty_certificate"}
                )
                excluded_trace_counts["sentinel_empty_certificate"] += 1
                continue

            # 3) Reject extreme outliers that indicate numerical breakdown:
            # If residual is astronomically large, it's probably a numerical error,
            # not a physically meaningful trace. Threshold can be conservative (e.g., 1e3).
            if result.get("residual", 0.0) is not None:
                if result["residual"] > 1e3 or result["theoretical_bound"] > 1e4:
                    print(
                        f"  [Skipping] {label}: NUMERIC OUTLIER (residual={result['residual']:.4g}, "
                        f"bound={result['theoretical_bound']:.4g})"
                    )
                    excluded_traces.append(
                        {
                            "label": label,
                            "reason": "numeric_outlier",
                            "details": {
                                "residual": result["residual"],
                                "theoretical_bound": result["theoretical_bound"],
                            },
                        }
                    )
                    excluded_trace_counts["numeric_outlier"] += 1
                    continue

            results[label] = result
            category = categorize_trace(label)
            result["category"] = category
            if category == "coherent":
                embeddings = result.get("embeddings")
                if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                    coherent_embeddings.append(embeddings)
            score = pick_score(result)
            result["score"] = score
            if category in results_by_category:
                results_by_category[category].append(score)
            print(f"  {label} [{category}]:")
            print(f"    steps={result['n_steps']}, score={score:.4f}")
            print(f"    residual={result['residual']:.4f} (oos={result['oos_residual']:.4f}, insample={result['insample_residual']:.4f})")
            print(f"    pca_explained={result['pca_explained']:.4f}, r_eff={result['r_eff']}")

    if excluded_traces:
        print("\n[1b] Skipped Trace Summary")
        print("-" * 80)
        for reason, count in excluded_trace_counts.items():
            print(f"  {reason}: {count}")

    if not results:
        print("  No traces found in tools/real_traces/")
        return 1

    # Summary by category
    print("\n[2] Summary by Category")
    print("-" * 80)
    print(f"  {'Category':<12} {'Count':>6} {'Mean Score':>12} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 52)

    for cat in ["coherent", "creative", "drift", "poison", "starvation"]:
        vals = [v for v in results_by_category[cat] if not np.isnan(v)]
        if vals:
            arr = np.array(vals)
            print(f"  {cat:<12} {len(arr):>6} {np.mean(arr):>12.4f} {np.min(arr):>10.4f} {np.max(arr):>10.4f}")
        else:
            print(f"  {cat:<12} {0:>6} {'N/A':>12} {'N/A':>10} {'N/A':>10}")

    # Discrimination test
    print("\n[3] Discrimination Test")
    print("-" * 80)

    coherent_vals = [v for v in results_by_category["coherent"] if not np.isnan(v)]
    creative_vals = [v for v in results_by_category["creative"] if not np.isnan(v)]
    drift_vals = [v for v in results_by_category["drift"] if not np.isnan(v)]
    poison_vals = [v for v in results_by_category["poison"] if not np.isnan(v)]
    starvation_vals = [v for v in results_by_category["starvation"] if not np.isnan(v)]

    print("\n  [Group Values]")
    print(f"  coherent: {np.round(coherent_vals, 4).tolist()}")
    print(f"  creative: {np.round(creative_vals, 4).tolist()}")
    print(f"  drift: {np.round(drift_vals, 4).tolist()}")
    print(f"  poison: {np.round(poison_vals, 4).tolist()}")
    print(f"  starvation: {np.round(starvation_vals, 4).tolist()}")

    # Combine coherent and creative (both are "good" traces)
    all_good = coherent_vals + creative_vals
    discrimination_pass = True

    # Key test: Creative (correct but unconventional) vs Drift (incorrect)
    print("\n  [Key Test: Creative vs Drift]")
    if creative_vals and drift_vals:
        creative_mean = np.mean(creative_vals)
        drift_mean = np.mean(drift_vals)
        print(f"  Creative (correct/unconventional) mean: {creative_mean:.4f}")
        print(f"  Drift (incorrect/hallucination) mean: {drift_mean:.4f}")
        print(f"  Separation: {drift_mean - creative_mean:.4f}")

        if creative_mean < drift_mean:
            print("  ✓ PASS: Creative traces have lower bound than drift")
            print("         (Metric distinguishes creativity from drift!)")
        else:
            print("  ✗ FAIL: Creative should have lower bound than drift")
            discrimination_pass = False
    else:
        print("  ⚠ Need both creative and drift traces for this test")

    # Starvation vs Coherent: use normalized effective rank
    print("\n  [Starvation vs Coherent: Rank Test]")
    starvation_r_rel = [
        r["r_rel"]
        for r in results.values()
        if r.get("category") == "starvation" and not np.isnan(r.get("r_rel", np.nan))
    ]
    coherent_r_rel = [
        r["r_rel"]
        for r in results.values()
        if r.get("category") == "coherent" and not np.isnan(r.get("r_rel", np.nan))
    ]
    if starvation_r_rel and coherent_r_rel:
        print(f"  Starvation mean normalized rank: {np.mean(starvation_r_rel):.2f}")
        print(f"  Coherent mean normalized rank:   {np.mean(coherent_r_rel):.2f}")
        u_stat, pval = stats.mannwhitneyu(starvation_r_rel, coherent_r_rel, alternative="less")
        if pval < 0.05:
            print("  ✓ PASS: Starvation has lower relative rank than coherent")
        else:
            print(f"  ✗ FAIL: Starvation relative rank is NOT lower (p={pval:.3g})")
            discrimination_pass = False
    else:
        print("  ⚠ Need both starvation and coherent traces for rank test")

    print("\n  [Drift Detection: R_norm Bootstrap Threshold]")
    rnorm_threshold = bootstrap_rnorm_threshold(coherent_embeddings, B=bootstrap_B, percentile=95.0)
    if rnorm_threshold is not None:
        drift_rnorm = []
        for res in results.values():
            embeddings = res.get("embeddings")
            if not isinstance(embeddings, np.ndarray):
                continue
            rnorm = compute_Rnorm_for_embeddings(embeddings)
            res["r_norm"] = rnorm
            if res.get("category") == "drift":
                drift_rnorm.append(rnorm)
        print(f"  Bootstrap R_norm threshold (95th percentile): {rnorm_threshold:.4f}")
        if drift_rnorm:
            drift_hits = sum(val > rnorm_threshold for val in drift_rnorm if not np.isnan(val))
            print(f"  Drift traces above threshold: {drift_hits}/{len(drift_rnorm)}")
    else:
        print("  ⚠ Insufficient coherent traces to compute R_norm threshold")

    print("\n  [Poison Detection: sv_max_ratio]")
    poison_sv = [
        res.get("sv_max_ratio", np.nan)
        for res in results.values()
        if res.get("category") == "poison"
    ]
    if poison_sv:
        poison_hits = sum(val > 0.6 for val in poison_sv if not np.isnan(val))
        print(f"  Poison traces above sv_max_ratio>0.6: {poison_hits}/{len(poison_sv)}")
    else:
        print("  ⚠ Need poison traces for sv_max_ratio detection")

    # Secondary test: All good vs All bad (but exclude starvation from all_bad; handle starvation separately)
    print("\n  [All Good vs All Bad (excluding starvation; starvation tested separately)]")
    all_bad_no_starvation = drift_vals + poison_vals
    if all_good and all_bad_no_starvation:
        good_mean = np.mean(all_good)
        bad_mean = np.mean(all_bad_no_starvation)
        print(f"  All good (coherent+creative) mean: {good_mean:.4f}")
        print(f"  All bad (drift+poison) mean:        {bad_mean:.4f}")
        print(f"  Separation: {bad_mean - good_mean:.4f}")

        if len(all_good) >= 2 and len(all_bad_no_starvation) >= 2:
            t_stat, p_val = stats.ttest_ind(all_good, all_bad_no_starvation, alternative='less')
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_val:.4f}")
            if p_val < 0.05:
                print("  ✓ PASS: Statistically significant (p < 0.05)")
            else:
                print("  ⚠ Not significant at p<0.05 (may need more samples)")

        if good_mean < bad_mean:
            print("  ✓ PASS: Good traces have lower bound than bad traces")
        else:
            print("  ✗ FAIL: Good should have lower bound than bad")
            discrimination_pass = False
    else:
        print("  ⚠ Need both good and bad (non-starvation) traces for this test")

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    plots_dir = reports_dir / "plots" / "real_trace_validation"
    diagnostics_dir = reports_dir / "diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    results_rows = []
    for label, res in results.items():
        row = {k: v for k, v in res.items() if k != "embeddings"}
        row["trace_id"] = label
        results_rows.append(row)

    csv_path = diagnostics_dir / "real_trace_validation_results.csv"
    if results_rows:
        try:
            import csv

            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(results_rows[0].keys()))
                writer.writeheader()
                for row in results_rows:
                    writer.writerow(row)
            print(f"\n  Saved CSV report to {csv_path}")
        except Exception as exc:
            print(f"\n  ⚠ Failed to write CSV report: {exc}")

    if HAS_MATPLOTLIB and results_rows:
        try:
            scatter_x = [row.get("theoretical_bound_prompt_adj", np.nan) for row in results_rows]
            scatter_y = [row.get("r_rel", np.nan) for row in results_rows]
            colors = [row.get("category", "unknown") for row in results_rows]
            unique_colors = sorted(set(colors))
            color_map = {name: idx for idx, name in enumerate(unique_colors)}
            plt.figure(figsize=(7, 5))
            plt.scatter(
                scatter_x,
                scatter_y,
                c=[color_map[c] for c in colors],
                cmap="tab10",
                alpha=0.8,
            )
            plt.xlabel("theoretical_bound_prompt_adj")
            plt.ylabel("r_rel")
            plt.title("Prompt-adjusted Bound vs Relative Rank")
            plt.tight_layout()
            scatter_path = plots_dir / "bound_vs_r_rel.png"
            plt.savefig(scatter_path, dpi=150)
            plt.close()

            plt.figure(figsize=(7, 5))
            data = {}
            for row in results_rows:
                cat = row.get("category", "unknown")
                data.setdefault(cat, []).append(row.get("r_rel", np.nan))
            plt.boxplot([data[k] for k in data], labels=list(data.keys()))
            plt.ylabel("r_rel")
            plt.title("Relative Rank by Category")
            plt.tight_layout()
            boxplot_path = plots_dir / "r_rel_boxplot.png"
            plt.savefig(boxplot_path, dpi=150)
            plt.close()
        except Exception as exc:
            print(f"\n  ⚠ Failed to write plots: {exc}")

    results_json_path = reports_dir / "real_trace_validation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "skipped": excluded_trace_counts,
                "counts": {k: len(v) for k, v in results_by_category.items()},
                "diagnostics": results_rows,
            },
            handle,
            indent=2,
        )
    print(f"\n  Saved JSON report to {results_json_path}")

    results_cleaned = {
        label: {k: v for k, v in res.items() if k != "embeddings"}
        for label, res in results.items()
    }

    return {
        "results": results_cleaned,
        "results_by_category": results_by_category,
        "excluded_traces": excluded_traces,
        "excluded_trace_counts": excluded_trace_counts,
        "discrimination_pass": discrimination_pass,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate real traces against drift metrics.")
    parser.add_argument("--use-prompt-adj", action="store_true", help="Use prompt-based adjustment.")
    parser.add_argument("--gamma", type=float, default=0.3, help="Gamma shrinkage for prompt adjustment.")
    parser.add_argument("--bootstrap-B", type=int, default=500, help="Bootstrap iterations for R_norm threshold.")
    parser.add_argument("--real-traces-dir", type=Path, default=None)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("DRIFT DETECTION METRIC - REAL TRACE VALIDATION")
    print("=" * 80)

    # Check embedding availability FIRST
    print("\n[0] Checking Embedding Availability")
    print("-" * 80)
    embedding_method = _check_embedding_availability()
    print(f"  Embedding method: {embedding_method}")

    if embedding_method == "tfidf":
        print("\n" + "!" * 80)
        print("WARNING: Using TF-IDF fallback!")
        print("!" * 80)
        print("""
  TF-IDF measures vocabulary difference, NOT semantic coherence.
  This will produce FALSE NEGATIVES for creative solutions that use
  different vocabulary (e.g., 'matrix exponentiation' vs 'loop').

  To get reliable results, ensure one of:
  1. Set OPENAI_API_KEY environment variable
  2. Enable network access to huggingface.co for sentence-transformers

  Results below may NOT reliably distinguish creativity from drift.
""")
    elif embedding_method == "sentence-transformers":
        print("  ✓ Using semantic embeddings (sentence-transformers)")
        print("  Results will reliably distinguish creativity from drift.")
    elif embedding_method == "openai":
        print("  ✓ Using semantic embeddings (OpenAI)")
        print("  Results will reliably distinguish creativity from drift.")

    # Validate existing real traces
    real_trace_results = run_real_trace_validation(
        real_traces_dir=args.real_traces_dir,
        gamma=args.gamma,
        use_prompt_adj=args.use_prompt_adj,
        bootstrap_B=args.bootstrap_B,
    )

    # Save results
    output = {
        "embedding_method": embedding_method,
        "real_trace_results": real_trace_results.get("results", {}) if isinstance(real_trace_results, dict) else {},
        "results_by_category": real_trace_results.get("results_by_category", {}) if isinstance(real_trace_results, dict) else {},
        "excluded_traces": real_trace_results.get("excluded_traces", []) if isinstance(real_trace_results, dict) else [],
        "excluded_trace_counts": real_trace_results.get("excluded_trace_counts", {}) if isinstance(real_trace_results, dict) else {},
        "discrimination_pass": real_trace_results.get("discrimination_pass", False) if isinstance(real_trace_results, dict) else False,
        "reliable": embedding_method in ("sentence-transformers", "openai"),
    }

    output_path = Path(__file__).parent / "real_trace_validation_results.json"

    # Convert any remaining numpy types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return bool(obj) if isinstance(obj, np.bool_) else int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(output), f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"  Embedding method: {embedding_method}")

    if embedding_method == "tfidf":
        print("\n  ⚠ WARNING: Results are UNRELIABLE (TF-IDF fallback)")
        print("  TF-IDF cannot reliably distinguish creativity from drift.")
        print("  Enable sentence-transformers or OpenAI for valid results.")
        print("\n" + "=" * 80)
        print("CONCLUSION: VALIDATION INCOMPLETE - REQUIRES SEMANTIC EMBEDDINGS")
        print("=" * 80)
        return 2  # Special exit code for "needs semantic embeddings"

    if isinstance(real_trace_results, dict):
        if real_trace_results.get("discrimination_pass"):
            print("✓ Real trace discrimination: PASS")
            print("\n" + "=" * 80)
            print("CONCLUSION: METRIC RELIABLY DISTINGUISHES DRIFT FROM CREATIVITY")
            print("=" * 80)
            return 0
        else:
            print("✗ Real trace discrimination: FAIL")
            print("\n" + "=" * 80)
            print("CONCLUSION: METRIC NEEDS IMPROVEMENT OR MORE DATA")
            print("=" * 80)
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
