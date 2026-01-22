import numpy as np
import pytest

from tools.feature_utils import NormalizationConfig, compute_trace_features

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_compute_trace_features_returns_new_keys():
    X = np.random.randn(12, 64)
    normalization = NormalizationConfig(l2_normalize_steps=True, trim_proportion=0.0)
    res = compute_trace_features(X, normalization=normalization)
    for key in ("residual_norm", "residual_fro_norm", "theoretical_bound_norm", "r_rel", "sv_max_ratio"):
        assert key in res
