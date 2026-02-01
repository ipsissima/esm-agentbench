"""Sensitivity tests for numerical thresholds.

This module tests the impact of different threshold values on certificate
generation, helping users understand the trade-offs between conservatism
and permissiveness.
"""
import numpy as np
import pytest
from typing import List, Tuple

from tests.test_guards import unit


@unit
class TestThresholdSensitivity:
    """Test sensitivity of certificate generation to threshold parameters."""

    def test_oos_residual_floor_sensitivity(self):
        """Test impact of different OOS residual floor values."""
        from certificates.make_certificate import _compute_oos_residual

        # Short trace (should use floor)
        Z_short = np.random.randn(3, 2)

        floors = [0.01, 0.05, 0.1, 0.2, 0.5]
        results = []

        for floor in floors:
            res, valid = _compute_oos_residual(Z_short, residual_floor=floor)
            results.append((floor, res, valid))
            
            # Short trace should always use floor
            assert res == floor, f"Expected floor {floor}, got {res}"
            assert valid is False, "Short trace should have oos_valid=False"

    def test_oos_validation_k_sensitivity(self):
        """Test impact of different OOS validation K values."""
        from certificates.make_certificate import _compute_oos_residual

        # Normal trace
        np.random.seed(42)
        Z = np.random.randn(20, 4)

        k_values = [1, 2, 3, 5, 8]
        results = []

        for k in k_values:
            res, valid = _compute_oos_residual(Z, oos_k=k)
            results.append((k, res, valid))
            
            # Should succeed for all K values with T=20
            assert valid is True, f"K={k} should be valid for T=20"
            assert res >= 0, f"Residual should be non-negative"

        # Residuals should be similar but not identical
        residuals = [r for k, r, v in results]
        assert max(residuals) / min(residuals) < 5.0, "K values should produce similar residuals"

    def test_witness_condition_threshold_validation(self):
        """Test witness condition number validation."""
        # This is a placeholder for when witness validation is integrated
        # with the config system
        from esmassessor.config import get_certificate_config, reset_config
        import os

        # Test default
        reset_config()
        config = get_certificate_config()
        assert config.witness_condition_number_threshold == 1e8

        # Test custom value
        reset_config()
        os.environ["WITNESS_COND_THRESHOLD"] = "1e6"
        config = get_certificate_config()
        assert config.witness_condition_number_threshold == 1e6
        
        # Cleanup
        del os.environ["WITNESS_COND_THRESHOLD"]
        reset_config()

    def test_explained_variance_threshold_validation(self):
        """Test explained variance threshold validation."""
        from esmassessor.config import get_certificate_config, reset_config
        import os

        # Test default
        reset_config()
        config = get_certificate_config()
        assert config.explained_variance_threshold == 0.90

        # Test custom value
        reset_config()
        os.environ["EXPLAINED_VARIANCE_THRESHOLD"] = "0.95"
        config = get_certificate_config()
        assert config.explained_variance_threshold == 0.95
        
        # Cleanup
        del os.environ["EXPLAINED_VARIANCE_THRESHOLD"]
        reset_config()

    def test_all_thresholds_from_env(self):
        """Test that all threshold parameters can be loaded from environment."""
        from esmassessor.config import get_certificate_config, reset_config
        import os

        # Set all threshold environment variables
        env_vars = {
            "WITNESS_COND_THRESHOLD": "5e7",
            "WITNESS_GAP_THRESHOLD": "1e-5",
            "EXPLAINED_VARIANCE_THRESHOLD": "0.92",
            "OOS_VALIDATION_K": "5",
            "OOS_RESIDUAL_FLOOR": "0.15",
        }

        try:
            reset_config()
            for key, value in env_vars.items():
                os.environ[key] = value

            config = get_certificate_config()

            # Verify all values were loaded
            assert config.witness_condition_number_threshold == 5e7
            assert config.witness_gap_threshold == 1e-5
            assert config.explained_variance_threshold == 0.92
            assert config.oos_validation_k == 5
            assert config.oos_residual_floor == 0.15

        finally:
            # Cleanup
            for key in env_vars:
                if key in os.environ:
                    del os.environ[key]
            reset_config()

    def test_config_validation_ranges(self):
        """Test that config validation enforces valid ranges."""
        from esmassessor.config import CertificateConfig
        from pydantic import ValidationError

        # Valid config
        config = CertificateConfig(
            witness_condition_number_threshold=1e8,
            witness_gap_threshold=1e-6,
            explained_variance_threshold=0.90,
            oos_validation_k=3,
            oos_residual_floor=0.1,
        )
        assert config is not None

        # Invalid: condition number < 1
        with pytest.raises(ValidationError):
            CertificateConfig(witness_condition_number_threshold=0.5)

        # Invalid: explained variance > 1
        with pytest.raises(ValidationError):
            CertificateConfig(explained_variance_threshold=1.5)

        # Invalid: explained variance < 0
        with pytest.raises(ValidationError):
            CertificateConfig(explained_variance_threshold=-0.1)

        # Invalid: K < 1
        with pytest.raises(ValidationError):
            CertificateConfig(oos_validation_k=0)


@unit
class TestThresholdDocumentation:
    """Test that threshold documentation is complete and accurate."""

    def test_all_thresholds_documented(self):
        """Verify all thresholds are documented in NUMERICAL_THRESHOLDS.md."""
        from pathlib import Path

        docs_path = Path(__file__).parent.parent / "docs" / "NUMERICAL_THRESHOLDS.md"
        assert docs_path.exists(), "NUMERICAL_THRESHOLDS.md not found"

        content = docs_path.read_text()

        # Check that all threshold parameters are documented
        required_params = [
            "witness_condition_number_threshold",
            "witness_gap_threshold",
            "explained_variance_threshold",
            "oos_validation_k",
            "oos_residual_floor",
        ]

        for param in required_params:
            assert param in content or param.upper() in content, \
                f"Parameter {param} not documented in NUMERICAL_THRESHOLDS.md"

        # Check that default values are mentioned
        assert "1e8" in content  # witness_condition_number_threshold default
        assert "1e-6" in content  # witness_gap_threshold default
        assert "0.90" in content or "90%" in content  # explained_variance_threshold default
        assert "K=3" in content or "Default:** `3`" in content  # oos_validation_k default
        assert "0.1" in content  # oos_residual_floor default

    def test_rationale_provided(self):
        """Verify that rationale is provided for each threshold."""
        from pathlib import Path

        docs_path = Path(__file__).parent.parent / "docs" / "NUMERICAL_THRESHOLDS.md"
        content = docs_path.read_text()

        # Each section should have a "Rationale:" subsection
        rationale_count = content.count("**Rationale:**")
        assert rationale_count >= 5, "Each threshold should have a documented rationale"

    def test_sensitivity_guidance_provided(self):
        """Verify that sensitivity guidance is provided."""
        from pathlib import Path

        docs_path = Path(__file__).parent.parent / "docs" / "NUMERICAL_THRESHOLDS.md"
        content = docs_path.read_text()

        # Each section should have a "Sensitivity:" subsection
        sensitivity_count = content.count("**Sensitivity:**")
        assert sensitivity_count >= 5, "Each threshold should have sensitivity guidance"

        # Should mention recommended ranges
        assert "Recommended Range" in content or "recommended range" in content
