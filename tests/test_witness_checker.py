import numpy as np
import pytest

from certificates.witness_checker import WitnessValidationError, check_witness

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_check_witness_valid() -> None:
    np.random.seed(0)
    X0 = np.random.randn(3, 5)
    X1 = np.random.randn(3, 5)
    A = np.eye(3)
    diagnostics = check_witness(X0, X1, A, k=2)
    assert diagnostics["condition_number"] > 0
    assert diagnostics["spectral_gap"] >= 0
    assert diagnostics["relative_gap"] >= 0
    assert diagnostics["r_eff_checked"] == 2
    assert diagnostics["n_train_cols"] == 5


def test_check_witness_nan_rejected() -> None:
    X0 = np.array([[np.nan, 0.0, 1.0], [0.0, 1.0, 2.0]])
    X1 = np.zeros_like(X0)
    A = np.eye(2)
    with pytest.raises(WitnessValidationError, match="non-finite"):
        check_witness(X0, X1, A, k=1)


def test_check_witness_ill_conditioned() -> None:
    X0 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    X1 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    A = np.eye(2)
    with pytest.raises(WitnessValidationError, match="Condition number"):
        check_witness(X0, X1, A, k=1)


def test_relative_gap_prevents_false_positive_with_tiny_values() -> None:
    """Test that relative gap threshold prevents false positives when all singular values are tiny."""
    # Create a matrix with very small but well-separated singular values
    # This should pass relative gap check but might fail absolute gap check
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = np.array([1e-8, 1e-9])  # Small values with 10x separation
    Vt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    X0 = U @ np.diag(S) @ Vt
    X1 = X0.copy()
    A = np.eye(2)
    
    # With only absolute gap threshold, this would fail (gap = 9e-9 < 1e-12 is False, but close)
    # With relative gap threshold, this should pass (relative_gap = 0.9 > 1e-6)
    diagnostics = check_witness(X0, X1, A, k=1, gap_thresh=1e-12, relative_gap_thresh=1e-6)
    
    # Verify the relative gap is computed correctly
    assert diagnostics["relative_gap"] > 0.8  # Should be ~0.9


def test_relative_gap_fails_when_both_thresholds_violated() -> None:
    """Test that witness check fails when both absolute and relative gaps are too small."""
    # Create a matrix with nearly identical singular values
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = np.array([1.0, 0.999999])  # Very close singular values
    Vt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    X0 = U @ np.diag(S) @ Vt
    X1 = X0.copy()
    A = np.eye(2)
    
    # Both absolute (1e-6) and relative (1e-6) gaps should be too small
    with pytest.raises(WitnessValidationError, match="gap too small"):
        check_witness(X0, X1, A, k=1, gap_thresh=1e-5, relative_gap_thresh=1e-5)


def test_relative_gap_passes_with_good_separation() -> None:
    """Test that witness check passes when singular values are well separated."""
    # Create a matrix with well-separated singular values
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = np.array([10.0, 1.0])  # Good separation (gap=9, relative=0.9)
    Vt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    X0 = U @ np.diag(S) @ Vt
    X1 = X0.copy()
    A = np.eye(2)
    
    diagnostics = check_witness(X0, X1, A, k=1, gap_thresh=1e-12, relative_gap_thresh=1e-6)
    
    assert diagnostics["spectral_gap"] > 8.9
    assert diagnostics["relative_gap"] > 0.8


def test_relative_gap_edge_case_single_singular_value() -> None:
    """Test behavior when there's only one singular value (gap = inf)."""
    # Rank-1 matrix
    X0 = np.array([[1.0, 2.0, 3.0]])
    X1 = np.array([[2.0, 4.0, 6.0]])
    A = np.array([[2.0]])
    
    diagnostics = check_witness(X0, X1, A, k=1, gap_thresh=1e-12, relative_gap_thresh=1e-6)
    
    # Gap should be infinite when there's only one singular value
    assert diagnostics["spectral_gap"] == float("inf")
    # relative_gap should also be infinite
    assert diagnostics["relative_gap"] == float("inf")
