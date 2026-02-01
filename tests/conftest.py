"""Pytest configuration and fixtures for ESM-AgentBench tests.

This module provides test fixtures and mocks for running tests without
the verified kernel (which requires Coq/OCaml build).

Test Tiers:
- Tier 1 (unit): Fast tests with no native code or model loading
- Tier 2 (integration): Container tests with kernel and small models
- Tier 3 (heavy): GPU tests with full models and attestation

Environment Variables:
- ESM_ALLOW_KERNEL_LOAD: Allow kernel loading (default: 1)
- ESM_ALLOW_MODEL_LOAD: Allow model loading (default: 1)
- ESM_STRICT: Enable strict invariant checking (default: 0)
- ESM_KERNEL_SERVICE: Enable kernel-as-service mode (default: 0)
"""
import json
import os
import re
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from tools.real_agents_hf.trace import RUN_GENERATOR, create_index
from tests.test_guards import (
    pytest_configure,
    pytest_collection_modifyitems,
    is_kernel_allowed,
    is_model_allowed,
    TestTierManager,
)

# Repository root and key directories
REPO_ROOT = Path(__file__).parent.parent
REAL_TRACES_DIR = REPO_ROOT / "tools" / "real_traces"
SUBMISSIONS_DIR = REPO_ROOT / "submissions" / "ipsissima"

# Expected scenarios for submissions
EXPECTED_SCENARIOS = [
    "code_backdoor_injection",
    "supply_chain_poisoning",
    "test_oracle_manipulation",
    "code_review_bypass",
    "debug_credential_leak",
    "refactor_vuln_injection",
]


def _setup_submission_traces():
    """Set up the submissions directory structure from real traces.

    This copies traces from tools/real_traces/ into the expected
    submissions/ipsissima/{scenario}/experiment_traces_real_hf/ structure.
    """
    if not REAL_TRACES_DIR.exists():
        return

    # Parse trace filenames: {label}_{model}_{idx}_{timestamp}.json
    trace_pattern = re.compile(r'^(gold|creative|drift|poison|starvation)_(.+?)_(\d+)_(\d+)\.json$')

    # Organize traces by label and model
    traces_by_label_model = {}
    for trace_file in REAL_TRACES_DIR.glob("*.json"):
        match = trace_pattern.match(trace_file.name)
        if match:
            label, model, idx, timestamp = match.groups()
            key = (label, model)
            if key not in traces_by_label_model:
                traces_by_label_model[key] = []
            traces_by_label_model[key].append(trace_file)

    # Copy traces to each scenario's submissions directory
    for scenario in EXPECTED_SCENARIOS:
        scenario_traces_dir = SUBMISSIONS_DIR / scenario / "experiment_traces_real_hf"
        scenario_traces_dir.mkdir(parents=True, exist_ok=True)

        for (label, model), trace_files in traces_by_label_model.items():
            # Only use gold, creative, drift labels for scenarios
            if label not in ("gold", "creative", "drift"):
                continue

            target_dir = scenario_traces_dir / model / label
            target_dir.mkdir(parents=True, exist_ok=True)

            for idx, trace_file in enumerate(trace_files[:5]):  # Limit to 5 traces per label
                target_path = target_dir / f"run_{idx:03d}.json"
                if not target_path.exists():
                    shutil.copy2(trace_file, target_path)

        create_index(
            scenario_traces_dir,
            {
                "run_generator": RUN_GENERATOR,
                "test_fixture": True,
            },
        )


@pytest.fixture(scope="session", autouse=True)
def setup_submission_traces():
    """Ensure submission traces are set up before running tests."""
    _setup_submission_traces()
    yield
    # Cleanup is optional - leave traces in place for debugging


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure test environment at session start.

    This ensures that unit tests don't accidentally load the native kernel
    which could segfault if not properly built.

    The verified kernel loading is disabled by default in tests. Tests that
    need the real kernel should use the @pytest.mark.kernel marker and be
    run in integration/heavy tier CI.
    """
    # Disable kernel loading by default in tests
    # This prevents segfaults from missing/broken kernel builds
    if "ESM_ALLOW_KERNEL_LOAD" not in os.environ:
        os.environ["ESM_ALLOW_KERNEL_LOAD"] = "0"

    # Also set ESM_SKIP_VERIFIED_KERNEL to ensure fallback even if kernel loads
    if "ESM_SKIP_VERIFIED_KERNEL" not in os.environ:
        os.environ["ESM_SKIP_VERIFIED_KERNEL"] = "1"

    yield


@pytest.fixture(autouse=True)
def mock_verified_kernel(request):
    """Mock the verified kernel for tests that don't have kernel access.

    The verified kernel requires a compiled Coq/OCaml library that may not
    be available in all test environments. This fixture provides a Python
    fallback implementation for testing purposes.

    Tests marked with @pytest.mark.kernel that are run in integration/heavy
    tiers will use the real kernel (if available).
    """
    # Check if this specific test requires the real kernel
    has_kernel_marker = "kernel" in [marker.name for marker in request.node.iter_markers()]
    has_unit_marker = "unit" in [marker.name for marker in request.node.iter_markers()]

    # Use real kernel only if:
    # 1. Test has @pytest.mark.kernel marker
    # 2. Kernel loading is allowed
    # 3. We're in integration tier or higher
    # 4. Test is NOT marked as unit
    use_real_kernel = (
        has_kernel_marker and
        not has_unit_marker and
        is_kernel_allowed() and
        TestTierManager.current_tier() >= TestTierManager.TIER_INTEGRATION
    )

    if use_real_kernel:
        yield
        return

    def mock_kernel_compute_certificate(X0, X1, A, tail_energy, semantic_divergence, lipschitz_margin, strict=True):
        """Fallback Python implementation of kernel certificate computation.

        This is a pure Python implementation that mirrors the verified kernel's
        logic for testing purposes. In production, the verified kernel provides
        formal guarantees.
        """
        eps = 1e-12

        # Compute residual: ||X1 - A @ X0||_F / ||X1||_F
        err = X1 - A @ X0
        residual = float(np.linalg.norm(err, "fro") / (np.linalg.norm(X1, "fro") + eps))

        # Compute bound using constants
        c_res = 1.0
        c_tail = 1.0
        c_sem = 1.0
        c_robust = 1.0

        theoretical_bound = float(
            c_res * residual +
            c_tail * tail_energy +
            c_sem * semantic_divergence +
            c_robust * lipschitz_margin
        )

        return (residual, theoretical_bound)

    # Patch the kernel's compute_certificate function
    with patch('certificates.verified_kernel.compute_certificate', side_effect=mock_kernel_compute_certificate):
        # Also patch the import in make_certificate
        with patch('certificates.make_certificate.kernel_compute_certificate', side_effect=mock_kernel_compute_certificate):
            yield


@pytest.fixture
def kernel_adapter():
    """Provide a kernel adapter for tests.

    Returns the appropriate adapter based on test tier.
    """
    from esmassessor.kernel_adapter import make_kernel_adapter, reset_kernel_adapter

    # Reset to ensure clean state
    reset_kernel_adapter()

    adapter = make_kernel_adapter(prefer_verified=is_kernel_allowed())
    yield adapter

    # Cleanup
    reset_kernel_adapter()


@pytest.fixture
def runtime_policy():
    """Provide a runtime policy for tests."""
    from tools.real_agents_hf.runtime_policy import get_runtime_policy, reset_runtime_policy

    reset_runtime_policy()
    policy = get_runtime_policy()
    yield policy
    reset_runtime_policy()


@pytest.fixture
def strict_mode(monkeypatch):
    """Enable strict mode for a test."""
    monkeypatch.setenv("ESM_STRICT", "1")
    yield
    # monkeypatch auto-restores


@pytest.fixture
def unit_environment(monkeypatch):
    """Configure unit test environment (no native code)."""
    monkeypatch.setenv("ESM_ALLOW_KERNEL_LOAD", "0")
    monkeypatch.setenv("ESM_ALLOW_MODEL_LOAD", "0")
    monkeypatch.setenv("ESM_STRICT", "0")
    yield


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_embeddings(rng):
    """Generate sample embeddings for testing."""
    return rng.normal(size=(30, 64))


@pytest.fixture
def sample_task_embedding(rng):
    """Generate a sample task embedding."""
    return rng.normal(size=64)
