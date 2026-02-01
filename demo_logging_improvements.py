"""Demonstration of numerical fallback logging improvements.

This script demonstrates the new logging capabilities for numerical fallbacks
and embedder provenance checks as implemented in response to the code review.

Run with: python demo_logging_improvements.py
"""
import logging
import numpy as np
from certificates.make_certificate import (
    _compute_oos_residual,
    _fit_temporal_operator_ridge,
)
from core.certificate import compute_certificate_from_trace

# Configure logging to show INFO and WARNING messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

print("=" * 80)
print("DEMONSTRATION: Numerical Fallback Logging")
print("=" * 80)

# Demo 1: Short trace logging
print("\n1. Demonstrating short trace logging (T < 4):")
print("-" * 80)
Z_short = np.random.randn(3, 5)  # Only 3 timesteps
residual = _compute_oos_residual(Z_short)
print(f"   Result: residual={residual}")

# Demo 2: Normal computation (no fallback)
print("\n2. Demonstrating normal computation (well-conditioned):")
print("-" * 80)
d = 5
n = 20
X0 = np.random.randn(d, n)
X1 = np.random.randn(d, n)
A = _fit_temporal_operator_ridge(X0, X1, regularization=1e-6)
print(f"   Result: Successfully computed operator with shape {A.shape}")

# Demo 3: Embedder provenance check - missing embedder_id
print("\n3. Demonstrating embedder provenance warning (missing embedder_id):")
print("-" * 80)
trace = {
    "embeddings": np.random.randn(10, 5),
}
task_embedding = np.random.randn(5)
cert = compute_certificate_from_trace(
    trace,
    task_embedding=task_embedding,
    embedder_id=None,  # Missing - should trigger warning
)
print(f"   Result: Certificate computed, theoretical_bound={cert['theoretical_bound']:.4f}")

# Demo 4: Embedder provenance check - with embedder_id
print("\n4. Demonstrating embedder provenance logging (with embedder_id):")
print("-" * 80)
cert = compute_certificate_from_trace(
    trace,
    task_embedding=task_embedding,
    embedder_id="sentence-transformers/all-MiniLM-L6-v2",
)
print(f"   Result: Certificate computed with embedder_id={cert.get('embedder_id')}")

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)
print("\nKey improvements demonstrated:")
print("  ✓ Numerical fallbacks are logged with diagnostic information")
print("  ✓ Short traces trigger informative logging")
print("  ✓ Embedder provenance is validated and logged")
print("  ✓ Embedder ID is included in certificate metadata")
print("\nThese improvements make CI failures visible and support verifiable certificates.")
