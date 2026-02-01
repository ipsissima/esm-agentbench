"""Tests for verified kernel integration.

These tests verify:
1. Kernel input export creates valid JSON
2. Prototype kernel runs and produces valid output
3. Certificate bundle creation works correctly
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


class TestKernelInputExport:
    """Tests for kernel input JSON export."""

    def test_export_kernel_input_creates_file(self):
        """Test that export_kernel_input creates a valid JSON file."""
        from certificates.make_certificate import export_kernel_input

        X_aug = np.random.randn(20, 129)  # T=20, D+1=129

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = export_kernel_input(
                X_aug=X_aug,
                trace_id="test-trace-123",
                output_path=output_path,
                embedder_id="test-embedder",
                rank=10,
                precision_bits=128,
            )

            # Check file exists
            assert os.path.exists(output_path)

            # Load and validate structure
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data["schema_version"] == "1.0"
            assert data["trace_id"] == "test-trace-123"
            assert data["metadata"]["embedder_id"] == "test-embedder"
            assert data["parameters"]["rank"] == 10
            assert data["parameters"]["precision_bits"] == 128

            # Check observable structure
            x_aug = data["observables"]["X_aug"]
            assert x_aug["rows"] == 20
            assert x_aug["cols"] == 129
            assert "data_matrix" in x_aug
            assert "sha256" in x_aug

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_export_and_load_kernel_input_roundtrip(self):
        """Test that export and load produces identical matrix."""
        from certificates.make_certificate import export_kernel_input, load_kernel_input

        X_orig = np.random.randn(15, 64)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_kernel_input(
                X_aug=X_orig,
                trace_id="roundtrip-test",
                output_path=output_path,
            )

            X_loaded, metadata = load_kernel_input(output_path)

            np.testing.assert_allclose(X_loaded, X_orig, rtol=1e-10)
            assert metadata["trace_id"] == "roundtrip-test"

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_export_with_precomputed_operator(self):
        """Test export with pre-computed Koopman operator."""
        from certificates.make_certificate import export_kernel_input

        X_aug = np.random.randn(20, 10)
        A = np.random.randn(5, 5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = export_kernel_input(
                X_aug=X_aug,
                trace_id="operator-test",
                output_path=output_path,
                A_precompute=A,
            )

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data["koopman_fit"] is not None
            assert "A_precompute" in data["koopman_fit"]
            assert data["koopman_fit"]["A_precompute"]["rows"] == 5
            assert data["koopman_fit"]["A_precompute"]["cols"] == 5

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestPrototypeKernel:
    """Tests for prototype kernel execution."""

    @pytest.fixture
    def kernel_input_file(self, tmp_path):
        """Create a temporary kernel input file."""
        from certificates.make_certificate import export_kernel_input

        X_aug = np.random.randn(15, 10)
        input_path = tmp_path / "kernel_input.json"

        export_kernel_input(
            X_aug=X_aug,
            trace_id="pytest-test",
            output_path=str(input_path),
            precision_bits=64,
        )

        return str(input_path)

    def test_prototype_kernel_runs(self, kernel_input_file, tmp_path):
        """Test that prototype kernel produces valid output."""
        import subprocess

        output_path = tmp_path / "kernel_output.json"

        # Find kernel script
        repo_root = Path(__file__).parent.parent
        kernel_script = repo_root / "kernel" / "prototype" / "prototype_kernel.py"

        if not kernel_script.exists():
            pytest.skip("Prototype kernel not found")

        result = subprocess.run(
            ["python", str(kernel_script), kernel_input_file, str(output_path), "--precision", "64"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Kernel may exit with 1 if checks fail, but should produce output
        assert output_path.exists(), f"Kernel output not created. stderr: {result.stderr}"

        with open(output_path, 'r') as f:
            output = json.load(f)

        # Check structure
        assert "schema_version" in output
        assert "computed" in output
        assert "checks" in output
        assert "provenance" in output

        # Check computed fields
        computed = output["computed"]
        assert "sigma" in computed
        assert "gamma" in computed
        assert "tail_energy" in computed
        assert "koopman" in computed
        assert "residuals" in computed
        assert "per_step" in computed

        # Check checks
        checks = output["checks"]
        assert "theoretical_bound" in checks
        assert "pass" in checks["theoretical_bound"]
        assert isinstance(checks["theoretical_bound"]["pass"], bool)

    def test_kernel_client_diagnostics(self):
        """Test kernel client diagnostics function."""
        from certificates.kernel_client import get_kernel_diagnostics

        diag = get_kernel_diagnostics()

        assert "kernel_image" in diag
        assert "docker_available" in diag
        assert "prototype_available" in diag


class TestCertificateBundle:
    """Tests for certificate bundle creation."""

    def test_create_bundle_basic(self, tmp_path):
        """Test basic bundle creation."""
        from certificates.cert_bundle import create_bundle, verify_bundle

        # Create test files
        trace_path = tmp_path / "trace.json"
        cert_path = tmp_path / "certificate.json"

        with open(trace_path, 'w') as f:
            json.dump({"trace": "test"}, f)
        with open(cert_path, 'w') as f:
            json.dump({"certificate": "test"}, f)

        # Create bundle
        bundle_dir = tmp_path / "bundle"
        create_bundle(
            str(bundle_dir),
            trace_path=str(trace_path),
            certificate_path=str(cert_path),
            embedder_id="test-embedder",
            kernel_mode="prototype",
        )

        # Check bundle contents
        assert (bundle_dir / "trace.json").exists()
        assert (bundle_dir / "certificate.json").exists()
        assert (bundle_dir / "metadata.json").exists()

        # Verify bundle
        result = verify_bundle(str(bundle_dir))
        assert result["valid"]
        assert result["checks"]["certificate.json_present"]

    def test_verify_bundle_hash_check(self, tmp_path):
        """Test that bundle verification catches modified files."""
        from certificates.cert_bundle import create_bundle, verify_bundle

        # Create bundle
        cert_path = tmp_path / "certificate.json"
        with open(cert_path, 'w') as f:
            json.dump({"original": "data"}, f)

        bundle_dir = tmp_path / "bundle"
        create_bundle(str(bundle_dir), certificate_path=str(cert_path))

        # Modify a file after bundle creation
        with open(bundle_dir / "certificate.json", 'w') as f:
            json.dump({"modified": "data"}, f)

        # Verification should fail
        result = verify_bundle(str(bundle_dir))
        assert not result["valid"]
        assert any("hash" in str(e).lower() or "mismatch" in str(e).lower() for e in result["errors"])


class TestComputeCertificateWithKernelExport:
    """Integration tests for compute_certificate with kernel export."""

    def test_compute_certificate_returns_expected_fields(self):
        """Test that compute_certificate returns all expected fields."""
        from certificates.make_certificate import compute_certificate

        X = np.random.randn(20, 128)
        cert = compute_certificate(X, r=10, kernel_strict=False)

        # Check required fields
        assert "theoretical_bound" in cert
        assert "residual" in cert
        assert "tail_energy" in cert
        assert "pca_explained" in cert
        assert "sigma_max" in cert
        assert "singular_gap" in cert

        # Check values are reasonable
        assert cert["theoretical_bound"] >= 0
        assert 0 <= cert["pca_explained"] <= 1
        assert 0 <= cert["tail_energy"] <= 1
