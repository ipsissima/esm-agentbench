#!/usr/bin/env python
"""Test: Does eigenvalue magnitude improve discrimination?

Hypothesis: Coherent/Creative traces have eigenvalues near 1 (stable dynamics),
while Drift traces have eigenvalues < 1 (decaying/chaotic dynamics).
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm, pinv
from sklearn.decomposition import PCA
from scipy import stats
from typing import Dict, List

N_RUNS = 500
N_STEPS = 12
EMBEDDING_DIM = 64

np.random.seed(42)


def compute_certificate_variants(embeddings: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute certificate with multiple eigenvalue-based formulas."""
    X = np.array(embeddings, dtype=float)
    T = X.shape[0]
    eps = 1e-12

    default = {"bound": 2.0, "max_eig": 0.0, "eig_product": 0.0, "eig_sum": 0.0}
    if T < 2 or X.size == 0:
        return {"current": default, "v1_max_eig": default, "v2_eig_product": default, "v3_stability": default}

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
        return {"current": default, "v1_max_eig": default, "v2_eig_product": default, "v3_stability": default}

    X0 = Z[:-1].T
    X1 = Z[1:].T

    gram = X0 @ X0.T + eps * np.eye(Z.shape[1])
    A = (X1 @ X0.T) @ pinv(gram)

    # Compute eigenvalues
    eigs = np.linalg.eigvals(A)
    mags = np.abs(eigs)
    mags_sorted = np.sort(mags)[::-1]

    max_eig = float(mags_sorted[0]) if mags_sorted.size else 0.0
    second_eig = float(mags_sorted[1]) if mags_sorted.size > 1 else 0.0
    eig_product = max_eig * second_eig
    eig_sum = max_eig + second_eig

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

    # V1: Penalize LOW max eigenvalue (unstable dynamics)
    # Creative/Coherent: λ₁ near 1 → small penalty
    # Drift: λ₁ < 1 → larger penalty
    max_eig_penalty = max(0.0, 1.0 - max_eig)  # 0 when λ₁=1, positive when λ₁<1
    v1_bound = residual + pca_tail_estimate + smooth_hallucination_penalty + max_eig_penalty * 0.5

    # V2: Penalize LOW eigenvalue product (both modes decaying)
    # Creative: λ₁*λ₂ ≈ 0.88 (both near 1)
    # Drift: λ₁*λ₂ ≈ 0.35 (both decaying)
    eig_product_penalty = max(0.0, 0.8 - eig_product)  # Penalize when product < 0.8
    v2_bound = residual + pca_tail_estimate + smooth_hallucination_penalty + eig_product_penalty * 0.5

    # V3: Stability bonus - reward eigenvalues close to 1
    # Distance from unit circle: |1 - λ|
    stability = 1.0 - abs(1.0 - max_eig)  # 1 when λ=1, 0 when λ=0 or λ=2
    stability_bonus = stability * 0.3  # Reduce bound when stable
    v3_bound = residual + pca_tail_estimate + smooth_hallucination_penalty - stability_bonus

    return {
        "current": {
            "bound": current_bound,
            "residual": residual,
            "tail": pca_tail_estimate,
            "smooth_penalty": smooth_hallucination_penalty,
            "max_eig": max_eig,
            "second_eig": second_eig,
            "eig_product": eig_product,
        },
        "v1_max_eig": {
            "bound": v1_bound,
            "penalty": max_eig_penalty,
        },
        "v2_eig_product": {
            "bound": v2_bound,
            "penalty": eig_product_penalty,
        },
        "v3_stability": {
            "bound": v3_bound,
            "bonus": stability_bonus,
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
    print("EIGENVALUE MAGNITUDE EXPERIMENT")
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

    # Eigenvalue analysis
    print("-" * 80)
    print("EIGENVALUE PROPERTIES BY TRACE TYPE")
    print("-" * 80)
    print(f"{'Type':<12} {'λ₁':>10} {'λ₂':>10} {'λ₁*λ₂':>10} {'|1-λ₁|':>10}")
    print("-" * 80)

    for trace_type in trace_types:
        certs = results[trace_type]
        mean_l1 = np.mean([c["current"]["max_eig"] for c in certs])
        mean_l2 = np.mean([c["current"]["second_eig"] for c in certs])
        mean_prod = np.mean([c["current"]["eig_product"] for c in certs])
        mean_dist = np.mean([abs(1.0 - c["current"]["max_eig"]) for c in certs])
        print(f"{trace_type:<12} {mean_l1:>10.4f} {mean_l2:>10.4f} {mean_prod:>10.4f} {mean_dist:>10.4f}")

    # Statistical tests
    print()
    print("-" * 80)
    print("STATISTICAL TESTS: Do eigenvalue properties discriminate?")
    print("-" * 80)

    metrics = [
        ("λ₁ (max eigenvalue)", lambda c: c["current"]["max_eig"]),
        ("λ₁*λ₂ (product)", lambda c: c["current"]["eig_product"]),
        ("|1-λ₁| (instability)", lambda c: abs(1.0 - c["current"]["max_eig"])),
    ]

    for metric_name, extractor in metrics:
        creative_vals = [extractor(c) for c in results["creative"]]
        drift_vals = [extractor(c) for c in results["drift"]]

        t_stat, p_val = stats.ttest_ind(creative_vals, drift_vals)
        creative_mean = np.mean(creative_vals)
        drift_mean = np.mean(drift_vals)

        print(f"\n  {metric_name}:")
        print(f"    Creative: {creative_mean:.4f} ± {np.std(creative_vals):.4f}")
        print(f"    Drift:    {drift_mean:.4f} ± {np.std(drift_vals):.4f}")
        print(f"    t={t_stat:.2f}, p={p_val:.2e}")
        if p_val < 0.001:
            direction = "higher" if creative_mean > drift_mean else "lower"
            print(f"    ✓ Creative has significantly {direction} {metric_name}")
        else:
            print(f"    ~ No significant difference")

    # Compare formula variants
    print()
    print("-" * 80)
    print("FORMULA COMPARISON")
    print("-" * 80)

    variants = ["current", "v1_max_eig", "v2_eig_product", "v3_stability"]
    variant_names = {
        "current": "Current",
        "v1_max_eig": "+ Max λ penalty",
        "v2_eig_product": "+ λ₁λ₂ penalty",
        "v3_stability": "- Stability bonus",
    }

    print(f"\n{'Variant':<20} ", end="")
    for t in trace_types:
        print(f"{t:>12}", end="")
    print()
    print("-" * 70)

    variant_means = {v: {} for v in variants}
    for variant in variants:
        print(f"{variant_names[variant]:<20} ", end="")
        for trace_type in trace_types:
            mean_bound = np.mean([c[variant]["bound"] for c in results[trace_type]])
            variant_means[variant][trace_type] = mean_bound
            print(f"{mean_bound:>12.4f}", end="")
        print()

    # Discrimination gaps
    print()
    print("-" * 80)
    print("DISCRIMINATION GAPS: Creative vs Drift")
    print("-" * 80)
    print(f"{'Variant':<25} {'Gap':>12} {'vs Current':>12}")
    print("-" * 50)

    current_gap = variant_means["current"]["drift"] - variant_means["current"]["creative"]
    for variant in variants:
        gap = variant_means[variant]["drift"] - variant_means[variant]["creative"]
        if variant == "current":
            print(f"{variant_names[variant]:<25} {gap:>12.4f} {'(baseline)':>12}")
        else:
            change = (gap - current_gap) / current_gap * 100
            print(f"{variant_names[variant]:<25} {gap:>12.4f} {change:>+11.1f}%")

    # Find best variant
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    best_variant = max(variants, key=lambda v: variant_means[v]["drift"] - variant_means[v]["creative"])
    best_gap = variant_means[best_variant]["drift"] - variant_means[best_variant]["creative"]

    print(f"\nBest variant: {variant_names[best_variant]}")
    print(f"Creative vs Drift gap: {best_gap:.4f}")

    if best_variant != "current":
        improvement = (best_gap - current_gap) / current_gap * 100
        print(f"Improvement over current: {improvement:+.1f}%")

        if improvement > 20:
            print("\n✓ SIGNIFICANT IMPROVEMENT FOUND")
        elif improvement > 5:
            print("\n~ MARGINAL IMPROVEMENT")
        else:
            print("\n~ NO MEANINGFUL IMPROVEMENT")
    else:
        print("\nCurrent formula is already optimal among tested variants")

    # Also check gold vs drift
    print()
    print("-" * 80)
    print("BONUS: Gold vs Drift gaps")
    print("-" * 80)
    for variant in variants:
        gap = variant_means[variant]["drift"] - variant_means[variant]["gold"]
        print(f"{variant_names[variant]:<25} {gap:>12.4f}")


if __name__ == "__main__":
    run_experiment()
