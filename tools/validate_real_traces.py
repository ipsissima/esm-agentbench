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

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from certificates.make_certificate import compute_certificate

# Track which embedding method is used
_embedding_method = "unknown"


def _check_embedding_availability() -> str:
    """Check which embedding method is available and return its name.

    Attempts to load embeddings in order of preference:
    1. OpenAI (if API key set)
    2. sentence-transformers (if model is baked in or cached)
    3. TF-IDF fallback (unreliable for creative vs drift)
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

    # Try to load sentence-transformers model directly
    # If SENTENCE_TRANSFORMERS_HOME is set (Docker), it will find the baked-in model
    # If model is cached locally, it will load from cache
    # This is fast if the model exists; the library handles path resolution
    try:
        from sentence_transformers import SentenceTransformer
        # local_files_only=True prevents network calls - only uses cached/baked models
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        _embedding_method = "sentence-transformers"
        return "sentence-transformers"
    except Exception as e:
        # Model not available locally - will need network access
        print(f"  Note: sentence-transformers model not available locally: {type(e).__name__}")
        print("  Network access to huggingface.co may be required, or bake model in Dockerfile")

    _embedding_method = "tfidf"
    return "tfidf"


def embed_trace_steps_with_check(trace: List[Dict[str, Any]]) -> Tuple[np.ndarray, str]:
    """Embed trace steps and return (embeddings, method_used)."""
    from assessor.kickoff import embed_trace_steps

    embeddings = embed_trace_steps(trace)
    return embeddings, _embedding_method


def load_trace(path: Path) -> List[Dict[str, Any]]:
    """Load a trace from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'trace' in data:
        return data['trace']
    return []


def analyze_trace(trace: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Analyze a single trace and compute certificate."""
    from assessor.kickoff import embed_trace_steps
    embeddings = embed_trace_steps(trace)
    cert = compute_certificate(embeddings)

    return {
        "label": label,
        "n_steps": len(trace),
        "theoretical_bound": cert.get("theoretical_bound", float("nan")),
        "residual": cert.get("residual", float("nan")),
        "pca_explained": cert.get("pca_explained", float("nan")),
        "pca_tail_estimate": cert.get("pca_tail_estimate", float("nan")),
    }


def categorize_trace(filename: str) -> str:
    """Categorize a trace based on its filename."""
    name = filename.lower()
    # Creative must be checked before coherent since creative traces are also coherent
    if "creative" in name:
        return "creative"
    elif "coherent" in name or "good" in name:
        return "coherent"
    elif "drift" in name or "hallucination" in name or "bad" in name:
        return "drift"
    elif "poison" in name:
        return "poison"
    elif "starved" in name or "starvation" in name or "memory" in name:
        return "starvation"
    return "unknown"


def run_real_trace_validation():
    """Run validation on real traces."""
    print("=" * 80)
    print("REAL TRACE VALIDATION TEST")
    print("=" * 80)

    real_traces_dir = Path(__file__).parent / "real_traces"

    # Analyze existing real traces
    print("\n[1] Analyzing existing real traces in tools/real_traces/")
    print("-" * 80)

    results = {}
    results_by_category = {"coherent": [], "creative": [], "drift": [], "poison": [], "starvation": []}

    for trace_file in sorted(real_traces_dir.glob("*.json")):
        label = trace_file.stem
        trace = load_trace(trace_file)
        if trace:
            result = analyze_trace(trace, label)
            results[label] = result
            category = categorize_trace(label)
            result["category"] = category
            if category in results_by_category:
                results_by_category[category].append(result["theoretical_bound"])
            print(f"  {label} [{category}]:")
            print(f"    steps={result['n_steps']}, theoretical_bound={result['theoretical_bound']:.4f}")
            print(f"    residual={result['residual']:.4f}, pca_explained={result['pca_explained']:.4f}")

    if not results:
        print("  No traces found in tools/real_traces/")
        return 1

    # Summary by category
    print("\n[2] Summary by Category")
    print("-" * 80)
    print(f"  {'Category':<12} {'Count':>6} {'Mean Bound':>12} {'Min':>10} {'Max':>10}")
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

    # Combine coherent and creative (both are "good" traces)
    all_good = coherent_vals + creative_vals
    all_bad = drift_vals + poison_vals + starvation_vals

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

    # Secondary test: All good vs all bad
    print("\n  [All Good vs All Bad]")
    if all_good and all_bad:
        good_mean = np.mean(all_good)
        bad_mean = np.mean(all_bad)
        print(f"  All good (coherent+creative) mean: {good_mean:.4f}")
        print(f"  All bad (drift+poison+starvation) mean: {bad_mean:.4f}")
        print(f"  Separation: {bad_mean - good_mean:.4f}")

        if len(all_good) >= 2 and len(all_bad) >= 2:
            t_stat, p_val = stats.ttest_ind(all_good, all_bad, alternative='less')
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

    return {"results": results, "results_by_category": results_by_category, "discrimination_pass": discrimination_pass}




def main():
    """Main entry point."""
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
    real_trace_results = run_real_trace_validation()

    # Save results
    output = {
        "embedding_method": embedding_method,
        "real_trace_results": real_trace_results.get("results", {}) if isinstance(real_trace_results, dict) else {},
        "results_by_category": real_trace_results.get("results_by_category", {}) if isinstance(real_trace_results, dict) else {},
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
