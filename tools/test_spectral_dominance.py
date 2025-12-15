#!/usr/bin/env python
"""Test: Does adding spectral dominance penalty improve discrimination?

Hypothesis: Coherent traces have dominant eigenvalue (high spectral gap),
while drift traces have competing modes (low spectral gap).
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


def compute_certificate_variants(embeddings: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute certificate with multiple formula variants for comparison."""
    X = np.array(embeddings, dtype=float)
    T = X.shape[0]
    eps = 1e-12

    if T < 2 or X.size == 0:
        return {
            "current": {"bound": 2.0},
            "with_spectral": {"bound": 2.0},
        }

    X_aug = np.concatenate([X, np.ones((T, 1))], axis=1)
    r_eff = min(3, max(1, T // 2), T - 1, X_aug.shape[1])

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
        return {
            "current": {"bound": 2.0},
            "with_spectral": {"bound": 2.0},
        }

    X0 = Z[:-1].T
    X1 = Z[1:].T

    gram = X0 @ X0.T + eps * np.eye(Z.shape[1])
    A = (X1 @ X0.T) @ pinv(gram)

    # Compute eigenvalues of Koopman operator
    eigs = np.linalg.eigvals(A)
    mags = np.abs(eigs)
    mags_sorted = np.sort(mags)[::-1]

    max_eig = float(mags_sorted[0]) if mags_sorted.size else 1.0
    second_eig = float(mags_sorted[1]) if mags_sorted.size > 1 else 0.0
    spectral_gap = max_eig - second_eig

    residual = float(norm(X1 - A @ X0, ord="fro") / (norm(X1, ord="fro") + eps))

    # Current smooth hallucination penalty
    log_info = float(np.log1p(information_density))
    max_log_info = float(np.log1p(10.0))
    normalized_info = np.clip(log_info / max_log_info, 0.0, 1.0)
    residual_complement = np.clip(1.0 - residual, 0.0, 1.0)
    info_complement = np.clip(1.0 - normalized_info, 0.0, 1.0)
    smooth_hallucination_penalty = residual_complement * info_complement

    # CURRENT formula
    current_bound = residual + pca_tail_estimate + smooth_hallucination_penalty

    # NEW: Spectral dominance penalty
    # High when no dominant mode (drift), low when one mode dominates (coherent)
    spectral_dominance = max_eig / (max_eig + second_eig + eps)  # 0.5 to ~1.0
    spectral_penalty = (1.0 - spectral_dominance) * 2  # 0 to 1, scaled

    # Formula with spectral penalty
    spectral_bound = residual + pca_tail_estimate + smooth_hallucination_penalty + spectral_penalty

    return {
        "current": {
            "bound": current_bound,
            "residual": residual,
            "tail": pca_tail_estimate,
            "smooth_penalty": smooth_hallucination_penalty,
        },
        "with_spectral": {
            "bound": spectral_bound,
            "residual": residual,
            "tail": pca_tail_estimate,
            "smooth_penalty": smooth_hallucination_penalty,
            "spectral_penalty": spectral_penalty,
            "spectral_dominance": spectral_dominance,
            "spectral_gap": spectral_gap,
            "max_eig": max_eig,
            "second_eig": second_eig,
        },
    }


def generate_gold_trace(n_steps: int = N_STEPS, dim: int = EMBEDDING_DIM) -> np.ndarray:
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
    embeddings = []
    vec = np.random.randn(dim) * 0.5
    for step in range(n_steps):
        jump = np.random.randn(dim) * 0.8
        vec = vec + jump
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def generate_loop_trace(n_steps: int = N_STEPS, dim: int = EMBEDDING_DIM) -> np.ndarray:
    base = np.random.randn(dim) * 0.5
    base = base / (norm(base) + 1e-12)
    embeddings = []
    for step in range(n_steps):
        vec = base + np.random.randn(dim) * 0.01
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def run_experiment():
    print("=" * 80)
    print("SPECTRAL DOMINANCE PENALTY EXPERIMENT")
    print("=" * 80)
    print(f"Runs: {N_RUNS}, Steps: {N_STEPS}, Dim: {EMBEDDING_DIM}")
    print()

    trace_types = {
        "gold": generate_gold_trace,
        "creative": generate_creative_trace,
        "drift": generate_drift_trace,
        "loop": generate_loop_trace,
    }

    results: Dict[str, List[Dict]] = {t: [] for t in trace_types}

    for run in range(N_RUNS):
        np.random.seed(42 + run * 100)
        for trace_type, generator in trace_types.items():
            embeddings = generator()
            cert = compute_certificate_variants(embeddings)
            results[trace_type].append(cert)

    # Analyze spectral properties
    print("-" * 80)
    print("SPECTRAL PROPERTIES BY TRACE TYPE")
    print("-" * 80)
    print(f"{'Type':<12} {'Dominance':>12} {'Gap':>12} {'λ₁':>12} {'λ₂':>12} {'Spec Penalty':>12}")
    print("-" * 80)

    for trace_type in trace_types:
        certs = results[trace_type]
        mean_dom = np.mean([c["with_spectral"]["spectral_dominance"] for c in certs])
        mean_gap = np.mean([c["with_spectral"]["spectral_gap"] for c in certs])
        mean_l1 = np.mean([c["with_spectral"]["max_eig"] for c in certs])
        mean_l2 = np.mean([c["with_spectral"]["second_eig"] for c in certs])
        mean_penalty = np.mean([c["with_spectral"]["spectral_penalty"] for c in certs])
        print(f"{trace_type:<12} {mean_dom:>12.4f} {mean_gap:>12.4f} {mean_l1:>12.4f} {mean_l2:>12.4f} {mean_penalty:>12.4f}")

    # Compare formulas
    print()
    print("-" * 80)
    print("BOUND COMPARISON: CURRENT vs WITH SPECTRAL PENALTY")
    print("-" * 80)
    print(f"{'Type':<12} {'Current':>12} {'+ Spectral':>12} {'Difference':>12}")
    print("-" * 80)

    current_means = {}
    spectral_means = {}

    for trace_type in trace_types:
        certs = results[trace_type]
        current_mean = np.mean([c["current"]["bound"] for c in certs])
        spectral_mean = np.mean([c["with_spectral"]["bound"] for c in certs])
        current_means[trace_type] = current_mean
        spectral_means[trace_type] = spectral_mean
        diff = spectral_mean - current_mean
        print(f"{trace_type:<12} {current_mean:>12.4f} {spectral_mean:>12.4f} {diff:>+12.4f}")

    # Key discrimination metrics
    print()
    print("-" * 80)
    print("DISCRIMINATION GAPS")
    print("-" * 80)

    gaps = [
        ("Creative vs Drift", "creative", "drift"),
        ("Gold vs Drift", "gold", "drift"),
        ("Loop penalty vs Gold", "loop", "gold"),
    ]

    print(f"{'Metric':<25} {'Current':>12} {'+ Spectral':>12} {'Improvement':>12}")
    print("-" * 80)

    improvements = []
    for name, type1, type2 in gaps:
        current_gap = current_means[type2] - current_means[type1]
        spectral_gap = spectral_means[type2] - spectral_means[type1]

        if name == "Loop penalty vs Gold":
            current_gap = current_means[type1] - current_means[type2]
            spectral_gap = spectral_means[type1] - spectral_means[type2]

        if current_gap != 0:
            improvement = (spectral_gap - current_gap) / abs(current_gap) * 100
        else:
            improvement = 0
        improvements.append(improvement)
        print(f"{name:<25} {current_gap:>12.4f} {spectral_gap:>12.4f} {improvement:>+11.1f}%")

    # Verdict
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    creative_drift_current = current_means["drift"] - current_means["creative"]
    creative_drift_spectral = spectral_means["drift"] - spectral_means["creative"]

    if creative_drift_spectral > creative_drift_current * 1.2:
        print("✓ HYPOTHESIS SUPPORTED: Spectral dominance improves discrimination")
    elif creative_drift_spectral < creative_drift_current * 0.8:
        print("✗ HYPOTHESIS REFUTED: Spectral dominance hurts discrimination")
    else:
        print("~ INCONCLUSIVE: Marginal effect")

    print()
    print(f"  Creative vs Drift gap:")
    print(f"    Current formula:      {creative_drift_current:.4f}")
    print(f"    With spectral penalty: {creative_drift_spectral:.4f}")
    print(f"    Change: {((creative_drift_spectral - creative_drift_current) / creative_drift_current * 100):+.1f}%")

    # Check if spectral properties actually differ by trace type
    print()
    print("-" * 80)
    print("STATISTICAL TEST: Does spectral dominance differ by trace type?")
    print("-" * 80)

    gold_dom = [c["with_spectral"]["spectral_dominance"] for c in results["gold"]]
    drift_dom = [c["with_spectral"]["spectral_dominance"] for c in results["drift"]]

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(gold_dom, drift_dom)

    print(f"  Gold dominance mean:  {np.mean(gold_dom):.4f} ± {np.std(gold_dom):.4f}")
    print(f"  Drift dominance mean: {np.mean(drift_dom):.4f} ± {np.std(drift_dom):.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.2e}")

    if p_val < 0.05 and np.mean(gold_dom) > np.mean(drift_dom):
        print("  ✓ Gold has significantly higher spectral dominance than drift")
    elif p_val < 0.05 and np.mean(gold_dom) < np.mean(drift_dom):
        print("  ✗ Drift has higher spectral dominance (unexpected!)")
    else:
        print("  ~ No significant difference in spectral dominance")


if __name__ == "__main__":
    run_experiment()
