import numpy as np
import pytest

from certificates.witness_checker import WitnessValidationError, check_witness


def test_check_witness_valid() -> None:
    np.random.seed(0)
    X0 = np.random.randn(3, 5)
    X1 = np.random.randn(3, 5)
    A = np.eye(3)
    diagnostics = check_witness(X0, X1, A, k=2)
    assert diagnostics["condition_number"] > 0
    assert diagnostics["spectral_gap"] >= 0
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
