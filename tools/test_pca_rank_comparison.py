#!/usr/bin/env python
"""Empirical test: Does increasing PCA rank improve discrimination?

Tests r=3 (current), r=6, r=8 to verify/refute the hypothesis that
higher rank helps creative traces more than drift traces.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm, pinv
from sklearn.decomposition import PCA
from typing import Dict, List

N_RUNS = 500
N_STEPS = 12
EMBEDDING_DIM = 64

np.random.seed(42)


def compute_certificate_with_rank(embeddings: np.ndarray, r_max: int) -> Dict[str, float]:
    """Compute certificate with configurable PCA rank limit."""
    X = np.array(embeddings, dtype=float)
    T = X.shape[0]
    eps = 1e-12

    if T < 2 or X.size == 0:
        return {"theoretical_bound": 2.0, "residual": 1.0, "pca_tail_estimate": 1.0,
                "information_density": eps, "pca_explained": 0.0}

    X_aug = np.concatenate([X, np.ones((T, 1))], axis=1)

    # KEY VARIABLE: r_max controls the PCA rank limit
    r_eff = min(r_max, max(1, T // 2), T - 1, X_aug.shape[1])

    pca = PCA(n_components=r_eff, svd_solver="auto", random_state=0)
    pca.fit(X_aug)
    Z = X_aug @ pca.components_.T

    pca_explained = float(np.clip(np.sum(pca.explained_variance_ratio_), 0.0, 1.0))
    pca_tail_estimate = float(max(0.0, 1.0 - pca_explained))

    if len(pca.explained_variance_) > 0:
        avg_pca_variance = float(np.mean(pca.explained_variance_))
    else:
        avg_pca_variance = eps

    z_energy = norm(Z, ord="fro") / np.sqrt(T)
    information_density = max(avg_pca_variance, z_energy, eps)

    if Z.shape[0] < 2:
        return {"theoretical_bound": 2.0, "residual": 1.0, "pca_tail_estimate": pca_tail_estimate,
                "information_density": information_density, "pca_explained": pca_explained}

    X0 = Z[:-1].T
    X1 = Z[1:].T

    gram = X0 @ X0.T + eps * np.eye(Z.shape[1])
    A = (X1 @ X0.T) @ pinv(gram)

    residual = float(norm(X1 - A @ X0, ord="fro") / (norm(X1, ord="fro") + eps))

    # Smooth hallucination penalty
    log_info = float(np.log1p(information_density))
    max_log_info = float(np.log1p(10.0))
    normalized_info = np.clip(log_info / max_log_info, 0.0, 1.0)
    residual_complement = np.clip(1.0 - residual, 0.0, 1.0)
    info_complement = np.clip(1.0 - normalized_info, 0.0, 1.0)
    smooth_hallucination_penalty = residual_complement * info_complement

    theoretical_bound = float(residual + pca_tail_estimate + smooth_hallucination_penalty)

    return {
        "pca_explained": pca_explained,
        "residual": residual,
        "pca_tail_estimate": pca_tail_estimate,
        "information_density": information_density,
        "theoretical_bound": theoretical_bound,
        "r_eff": r_eff,
    }


def generate_gold_trace(n_steps: int = N_STEPS, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Coherent, linearly-evolving trace."""
    base = np.random.randn(dim) * 0.5
    direction = np.random.randn(dim) * 0.1
    direction = direction / (norm(direction) + 1e-12)

    embeddings = []
    for step in range(n_steps):
        vec = base + direction * step * 0.1 + np.random.randn(dim) * 0.05
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def generate_creative_trace(n_steps: int = N_STEPS, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Creative but coherent trace with mode switches."""
    base = np.random.randn(dim) * 0.5
    modes = [np.random.randn(dim) * 0.3 for _ in range(3)]

    embeddings = []
    current_mode = 0
    for step in range(n_steps):
        mode_weight = 0.5 + 0.5 * np.cos(step * 0.5)
        vec = base + modes[current_mode] * mode_weight + np.random.randn(dim) * 0.1
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
        if step > 0 and step % 4 == 0 and current_mode < len(modes) - 1:
            current_mode += 1
    return np.array(embeddings)


def generate_drift_trace(n_steps: int = N_STEPS, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Drifting trace with high-variance random jumps."""
    embeddings = []
    vec = np.random.randn(dim) * 0.5

    for step in range(n_steps):
        jump = np.random.randn(dim) * 0.8  # HIGH variance noise
        vec = vec + jump
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def generate_loop_trace(n_steps: int = N_STEPS, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Loop trace - nearly constant (identity mapping)."""
    base = np.random.randn(dim) * 0.5
    base = base / (norm(base) + 1e-12)

    embeddings = []
    for step in range(n_steps):
        vec = base + np.random.randn(dim) * 0.01  # Tiny noise
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def run_comparison():
    """Run comparison across different PCA ranks."""
    print("=" * 80)
    print("PCA RANK COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Runs: {N_RUNS}, Steps: {N_STEPS}, Dim: {EMBEDDING_DIM}")
    print()

    ranks_to_test = [3, 6, 8]
    trace_types = {
        "gold": generate_gold_trace,
        "creative": generate_creative_trace,
        "drift": generate_drift_trace,
        "loop": generate_loop_trace,
    }

    # Collect results
    results: Dict[int, Dict[str, List[Dict]]] = {r: {t: [] for t in trace_types} for r in ranks_to_test}

    for run in range(N_RUNS):
        np.random.seed(42 + run * 100)

        for trace_type, generator in trace_types.items():
            embeddings = generator()

            for r_max in ranks_to_test:
                cert = compute_certificate_with_rank(embeddings, r_max)
                results[r_max][trace_type].append(cert)

    # Analyze results
    print("-" * 80)
    print("RESULTS BY PCA RANK")
    print("-" * 80)

    for r_max in ranks_to_test:
        print(f"\n### PCA Rank Limit = {r_max} ###")
        print(f"{'Type':<12} {'Bound':>10} {'Residual':>10} {'PCA Exp':>10} {'Info Den':>10} {'Tail':>10}")
        print("-" * 62)

        means = {}
        for trace_type in trace_types:
            certs = results[r_max][trace_type]
            mean_bound = np.mean([c["theoretical_bound"] for c in certs])
            mean_residual = np.mean([c["residual"] for c in certs])
            mean_pca_exp = np.mean([c["pca_explained"] for c in certs])
            mean_info = np.mean([c["information_density"] for c in certs])
            mean_tail = np.mean([c["pca_tail_estimate"] for c in certs])

            means[trace_type] = {
                "bound": mean_bound,
                "residual": mean_residual,
                "pca_explained": mean_pca_exp,
                "info_density": mean_info,
                "tail": mean_tail,
            }

            print(f"{trace_type:<12} {mean_bound:>10.4f} {mean_residual:>10.4f} {mean_pca_exp:>10.4f} {mean_info:>10.4f} {mean_tail:>10.4f}")

        # Key metrics
        creative_drift_gap = means["drift"]["bound"] - means["creative"]["bound"]
        gold_drift_gap = means["drift"]["bound"] - means["gold"]["bound"]
        loop_penalty = means["loop"]["bound"] - means["gold"]["bound"]

        print()
        print(f"  Creative vs Drift gap: {creative_drift_gap:+.4f} {'(Good!)' if creative_drift_gap > 0 else '(BAD!)'}")
        print(f"  Gold vs Drift gap:     {gold_drift_gap:+.4f} {'(Good!)' if gold_drift_gap > 0 else '(BAD!)'}")
        print(f"  Loop penalty vs Gold:  {loop_penalty:+.4f} {'(Good!)' if loop_penalty > 0 else '(BAD!)'}")

    # Summary comparison
    print()
    print("=" * 80)
    print("SUMMARY: EFFECT OF INCREASING PCA RANK")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'r=3':>12} {'r=6':>12} {'r=8':>12} {'Best':>8}")
    print("-" * 70)

    metrics = []
    for r in ranks_to_test:
        creative_drift = np.mean([c["theoretical_bound"] for c in results[r]["drift"]]) - \
                        np.mean([c["theoretical_bound"] for c in results[r]["creative"]])
        metrics.append(creative_drift)

    best_r = ranks_to_test[np.argmax(metrics)]
    print(f"{'Creative vs Drift gap':<25} {metrics[0]:>12.4f} {metrics[1]:>12.4f} {metrics[2]:>12.4f} {'r=' + str(best_r):>8}")

    metrics = []
    for r in ranks_to_test:
        gold_drift = np.mean([c["theoretical_bound"] for c in results[r]["drift"]]) - \
                    np.mean([c["theoretical_bound"] for c in results[r]["gold"]])
        metrics.append(gold_drift)

    best_r = ranks_to_test[np.argmax(metrics)]
    print(f"{'Gold vs Drift gap':<25} {metrics[0]:>12.4f} {metrics[1]:>12.4f} {metrics[2]:>12.4f} {'r=' + str(best_r):>8}")

    metrics = []
    for r in ranks_to_test:
        loop_penalty = np.mean([c["theoretical_bound"] for c in results[r]["loop"]]) - \
                      np.mean([c["theoretical_bound"] for c in results[r]["gold"]])
        metrics.append(loop_penalty)

    best_r = ranks_to_test[np.argmax(metrics)]
    print(f"{'Loop penalty vs Gold':<25} {metrics[0]:>12.4f} {metrics[1]:>12.4f} {metrics[2]:>12.4f} {'r=' + str(best_r):>8}")

    # Drift residual (should stay high)
    print()
    print(f"{'Drift residual':<25} ", end="")
    for r in ranks_to_test:
        drift_res = np.mean([c["residual"] for c in results[r]["drift"]])
        print(f"{drift_res:>12.4f} ", end="")
    print()

    print(f"{'Creative residual':<25} ", end="")
    for r in ranks_to_test:
        creative_res = np.mean([c["residual"] for c in results[r]["creative"]])
        print(f"{creative_res:>12.4f} ", end="")
    print()

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Check if increasing rank helps
    r3_gap = np.mean([c["theoretical_bound"] for c in results[3]["drift"]]) - \
             np.mean([c["theoretical_bound"] for c in results[3]["creative"]])
    r8_gap = np.mean([c["theoretical_bound"] for c in results[8]["drift"]]) - \
             np.mean([c["theoretical_bound"] for c in results[8]["creative"]])

    if r8_gap > r3_gap * 1.5:
        print("Hypothesis SUPPORTED: Increasing PCA rank improves discrimination")
    elif r8_gap < r3_gap * 0.8:
        print("Hypothesis REFUTED: Increasing PCA rank HURTS discrimination")
    else:
        print("Hypothesis INCONCLUSIVE: PCA rank has minimal effect")

    print(f"  r=3 Creative-Drift gap: {r3_gap:.4f}")
    print(f"  r=8 Creative-Drift gap: {r8_gap:.4f}")
    print(f"  Change: {((r8_gap - r3_gap) / r3_gap * 100):+.1f}%")


if __name__ == "__main__":
    run_comparison()
