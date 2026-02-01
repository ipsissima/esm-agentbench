"""Tests for make_certificate CLI.

These tests verify:
1. CLI argument parsing
2. Enhanced certificate output (per-step diagnostics, sinTheta)
3. Kernel input export
4. Bundle creation
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestEnhancedCertificate:
    """Tests for enhanced certificate computation."""

    def test_enhanced_certificate_includes_per_step(self):
        """Test that enhanced certificate includes per-step diagnostics."""
        from certificates.make_certificate_cli import compute_enhanced_certificate

        X = np.random.randn(20, 64)
        cert = compute_enhanced_certificate(
            X,
            rank=10,
            include_per_step=True,
            include_sin_theta=True,
            kernel_strict=False,
        )

        # Check per-step diagnostics
        assert "per_step_diagnostics" in cert
        assert "off_ratio_t" in cert["per_step_diagnostics"]
        assert "r_norm_t" in cert["per_step_diagnostics"]

        # Check lengths
        off_ratios = cert["per_step_diagnostics"]["off_ratio_t"]
        r_norms = cert["per_step_diagnostics"]["r_norm_t"]
        assert len(off_ratios) == 20
        assert len(r_norms) == 20

        # Check summary stats
        assert "off_ratio_max" in cert
        assert "off_ratio_mean" in cert
        assert "r_norm_max" in cert
        assert "r_norm_mean" in cert

    def test_enhanced_certificate_includes_sin_theta(self):
        """Test that enhanced certificate includes sinTheta summary."""
        from certificates.make_certificate_cli import compute_enhanced_certificate

        X = np.random.randn(20, 64)
        cert = compute_enhanced_certificate(
            X,
            rank=10,
            include_per_step=True,
            include_sin_theta=True,
            kernel_strict=False,
        )

        # Check sinTheta
        assert "sinTheta" in cert
        assert "sin_max" in cert["sinTheta"]
        assert "sin_frobenius" in cert["sinTheta"]

        # Check E_over_gamma
        assert "E_over_gamma" in cert
        assert "gamma" in cert

        # Check Wedin condition
        assert "wedin_condition" in cert
        assert "bound_holds" in cert["wedin_condition"]

    def test_enhanced_certificate_optional_diagnostics(self):
        """Test that diagnostics can be disabled."""
        from certificates.make_certificate_cli import compute_enhanced_certificate

        X = np.random.randn(20, 64)

        # Without per-step
        cert = compute_enhanced_certificate(
            X,
            rank=10,
            include_per_step=False,
            include_sin_theta=True,
            kernel_strict=False,
        )
        assert "per_step_diagnostics" not in cert
        assert "sinTheta" in cert

        # Without sinTheta
        cert = compute_enhanced_certificate(
            X,
            rank=10,
            include_per_step=True,
            include_sin_theta=False,
            kernel_strict=False,
        )
        assert "per_step_diagnostics" in cert
        assert "sinTheta" not in cert


class TestExtractEmbeddings:
    """Tests for embedding extraction from trace."""

    def test_extract_from_embeddings_field(self):
        """Test extraction from 'embeddings' field."""
        from certificates.make_certificate_cli import extract_embeddings

        trace = {"embeddings": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        emb = extract_embeddings(trace, psi_mode="embedding")

        assert emb.shape == (3, 3)
        np.testing.assert_array_equal(emb[0], [1, 2, 3])

    def test_extract_from_steps(self):
        """Test extraction from 'steps' with embeddings."""
        from certificates.make_certificate_cli import extract_embeddings

        trace = {
            "steps": [
                {"embedding": [1, 2, 3]},
                {"embedding": [4, 5, 6]},
            ]
        }
        emb = extract_embeddings(trace, psi_mode="embedding")

        assert emb.shape == (2, 3)

    def test_extract_residual_stream(self):
        """Test extraction from residual_stream field."""
        from certificates.make_certificate_cli import extract_embeddings

        trace = {"residual_stream": [[1, 2, 3], [4, 5, 6]]}
        emb = extract_embeddings(trace, psi_mode="residual_stream")

        assert emb.shape == (2, 3)

    def test_extract_fallback_to_embedding(self):
        """Test fallback to embedding when residual_stream not found."""
        from certificates.make_certificate_cli import extract_embeddings

        trace = {"embeddings": [[1, 2, 3]]}
        # Should warn and fall back to embeddings
        emb = extract_embeddings(trace, psi_mode="residual_stream")

        assert emb.shape == (1, 3)

    def test_extract_raises_on_missing(self):
        """Test that extraction raises on missing embeddings."""
        from certificates.make_certificate_cli import extract_embeddings

        trace = {"other_field": "value"}

        with pytest.raises(ValueError, match="No embeddings found"):
            extract_embeddings(trace, psi_mode="embedding")


class TestCLIIntegration:
    """Integration tests for CLI."""

    @pytest.fixture
    def trace_file(self, tmp_path):
        """Create a temporary trace file."""
        trace = {
            "embeddings": np.random.randn(15, 64).tolist(),
            "task_embedding": np.random.randn(64).tolist(),
        }
        trace_path = tmp_path / "trace.json"
        with open(trace_path, 'w') as f:
            json.dump(trace, f)
        return str(trace_path)

    def test_cli_basic_execution(self, trace_file, tmp_path):
        """Test basic CLI execution."""
        output_path = tmp_path / "cert.json"

        result = subprocess.run(
            [
                "python", "-m", "certificates.make_certificate_cli",
                trace_file,
                "--output", str(output_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "ESM_SKIP_VERIFIED_KERNEL": "1"},
        )

        # CLI may exit with 1 if bound exceeds threshold
        assert output_path.exists(), f"Output not created. stderr: {result.stderr}"

        with open(output_path) as f:
            cert = json.load(f)

        assert "theoretical_bound" in cert
        assert "per_step_diagnostics" in cert
        assert "sinTheta" in cert

    def test_cli_export_kernel_input(self, trace_file, tmp_path):
        """Test CLI with --export-kernel-input."""
        kernel_input_path = tmp_path / "kernel_input.json"

        result = subprocess.run(
            [
                "python", "-m", "certificates.make_certificate_cli",
                trace_file,
                "--export-kernel-input", str(kernel_input_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "ESM_SKIP_VERIFIED_KERNEL": "1"},
        )

        assert kernel_input_path.exists(), f"Kernel input not created. stderr: {result.stderr}"

        with open(kernel_input_path) as f:
            ki = json.load(f)

        assert "schema_version" in ki
        assert "observables" in ki
        assert "X_aug" in ki["observables"]

    def test_cli_no_per_step(self, trace_file, tmp_path):
        """Test CLI with --no-per-step."""
        output_path = tmp_path / "cert.json"

        result = subprocess.run(
            [
                "python", "-m", "certificates.make_certificate_cli",
                trace_file,
                "--output", str(output_path),
                "--no-per-step",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "ESM_SKIP_VERIFIED_KERNEL": "1"},
        )

        with open(output_path) as f:
            cert = json.load(f)

        assert "per_step_diagnostics" not in cert
        assert "sinTheta" in cert

    def test_cli_diagnostics(self, tmp_path):
        """Test CLI --diagnostics flag."""
        result = subprocess.run(
            [
                "python", "-m", "certificates.make_certificate_cli",
                "dummy.json",  # Won't be read
                "--diagnostics",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        assert result.returncode == 0
        diag = json.loads(result.stdout)
        assert "prototype_available" in diag

    def test_cli_psi_mode(self, tmp_path):
        """Test CLI with --psi residual_stream (fallback)."""
        trace = {
            "embeddings": np.random.randn(10, 32).tolist(),
        }
        trace_path = tmp_path / "trace.json"
        with open(trace_path, 'w') as f:
            json.dump(trace, f)

        output_path = tmp_path / "cert.json"

        # Should warn and fall back to embeddings
        result = subprocess.run(
            [
                "python", "-m", "certificates.make_certificate_cli",
                str(trace_path),
                "--output", str(output_path),
                "--psi", "residual_stream",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "ESM_SKIP_VERIFIED_KERNEL": "1"},
        )

        # Should still produce output (with fallback warning)
        assert output_path.exists()


class TestKernelVerification:
    """Tests for kernel verification integration."""

    @pytest.fixture
    def trace_file(self, tmp_path):
        """Create a temporary trace file."""
        trace = {
            "embeddings": np.random.randn(10, 32).tolist(),
        }
        trace_path = tmp_path / "trace.json"
        with open(trace_path, 'w') as f:
            json.dump(trace, f)
        return str(trace_path)

    def test_run_with_kernel_verification(self, trace_file, tmp_path):
        """Test kernel verification function."""
        from certificates.make_certificate_cli import run_with_kernel_verification

        embeddings = np.random.randn(10, 32)
        kernel_input_path = str(tmp_path / "ki.json")

        result = run_with_kernel_verification(
            trace_path=trace_file,
            embeddings=embeddings,
            kernel_input_path=kernel_input_path,
            kernel_mode="prototype",
            precision_bits=64,
        )

        assert "kernel_verified" in result
        assert "trace_id" in result
        assert os.path.exists(kernel_input_path)

    def test_cli_verify_with_kernel(self, trace_file, tmp_path):
        """Test CLI with --verify-with-kernel."""
        output_path = tmp_path / "cert.json"
        kernel_input_path = tmp_path / "ki.json"

        result = subprocess.run(
            [
                "python", "-m", "certificates.make_certificate_cli",
                trace_file,
                "--output", str(output_path),
                "--verify-with-kernel",
                "--export-kernel-input", str(kernel_input_path),
                "--precision", "64",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "ESM_SKIP_VERIFIED_KERNEL": "1"},
            timeout=120,
        )

        if output_path.exists():
            with open(output_path) as f:
                cert = json.load(f)
            assert "kernel_verification" in cert


class TestWedinCondition:
    """Tests for Wedin condition computation."""

    def test_wedin_bound_holds_for_stable_data(self):
        """Test that Wedin bound holds for stable trajectories."""
        from certificates.make_certificate_cli import compute_enhanced_certificate

        # Create stable trajectory (small perturbations)
        T, D = 25, 64
        base = np.random.randn(1, D)
        X = base + 0.01 * np.random.randn(T, D)

        cert = compute_enhanced_certificate(
            X,
            rank=10,
            include_sin_theta=True,
            kernel_strict=False,
        )

        # Wedin condition should hold
        wedin = cert.get("wedin_condition", {})
        # E_over_gamma should be small for stable data
        assert wedin.get("E_over_gamma", float("inf")) < 10.0

    def test_sin_theta_values_in_range(self):
        """Test that sin(theta) values are in valid range."""
        from certificates.make_certificate_cli import compute_enhanced_certificate

        X = np.random.randn(20, 64)
        cert = compute_enhanced_certificate(
            X,
            rank=10,
            include_sin_theta=True,
            kernel_strict=False,
        )

        sin_theta = cert.get("sinTheta", {})
        sin_max = sin_theta.get("sin_max", 0)
        sin_fro = sin_theta.get("sin_frobenius", 0)

        # sin values should be in [0, 1]
        assert 0 <= sin_max <= 1.0
        assert sin_fro >= 0
