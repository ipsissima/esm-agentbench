"""Client for invoking verified numeric kernels.

This module provides a client that executes the verified kernel
(either in Docker or locally) to compute interval-bounded certificate values.

Supported kernel modes:
- prototype: Python-based mpmath.iv for development and testing
- arb: Docker container with ARB library (production)
- mpfi: Docker container with MPFI library (alternative production)

Environment Variables:
- ESM_KERNEL_IMAGE: Docker image for production kernel (default: ipsissima/kernel:latest)
- ESM_KERNEL_LOCAL_PY: Path to local prototype kernel script for non-Docker testing
- ESM_KERNEL_TIMEOUT: Timeout for kernel execution in seconds (default: 300)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default Docker image for verified kernel
KERNEL_IMAGE = os.environ.get("ESM_KERNEL_IMAGE", "ipsissima/kernel:latest")

# Path to local prototype kernel for non-Docker development
KERNEL_LOCAL_PY = os.environ.get("ESM_KERNEL_LOCAL_PY", "")

# Timeout for kernel execution (seconds)
KERNEL_TIMEOUT = int(os.environ.get("ESM_KERNEL_TIMEOUT", "300"))


class KernelClientError(Exception):
    """Raised when kernel execution fails."""
    pass


def run_kernel(
    kernel_input_path: str,
    output_path: Optional[str] = None,
    precision_bits: int = 128,
    mode: str = "prototype",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute verified kernel and return output.

    Runs the verified kernel container (or local prototype) to compute
    interval-bounded certificate values from the input JSON.

    Parameters
    ----------
    kernel_input_path : str
        Path to kernel_input.json file.
    output_path : Optional[str]
        Path for kernel_output.json. If None, uses temp file.
    precision_bits : int
        Precision for interval arithmetic (e.g., 128, 160, 256).
    mode : str
        Kernel mode: "prototype", "arb", or "mpfi".
    timeout : Optional[int]
        Execution timeout in seconds. Default from ESM_KERNEL_TIMEOUT.

    Returns
    -------
    Dict[str, Any]
        Parsed kernel output JSON containing computed intervals.

    Raises
    ------
    KernelClientError
        If kernel execution fails or returns invalid output.
    """
    if timeout is None:
        timeout = KERNEL_TIMEOUT

    kernel_input_path = os.path.abspath(kernel_input_path)
    if not os.path.exists(kernel_input_path):
        raise KernelClientError(f"Kernel input file not found: {kernel_input_path}")

    # Determine output path
    use_temp = output_path is None
    if use_temp:
        tmpdir = tempfile.mkdtemp(prefix="esm_kernel_")
        output_path = os.path.join(tmpdir, "kernel_output.json")
    else:
        output_path = os.path.abspath(output_path)
        tmpdir = None

    try:
        # Try local prototype first if configured or mode is prototype
        if mode == "prototype" or KERNEL_LOCAL_PY:
            result = _run_prototype_kernel(
                kernel_input_path, output_path, precision_bits, timeout
            )
        else:
            result = _run_docker_kernel(
                kernel_input_path, output_path, precision_bits, mode, timeout
            )

        return result

    finally:
        # Clean up temp directory if used
        if use_temp and tmpdir and os.path.exists(tmpdir):
            try:
                shutil.rmtree(tmpdir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir: {e}")


def _run_prototype_kernel(
    input_path: str,
    output_path: str,
    precision_bits: int,
    timeout: int,
) -> Dict[str, Any]:
    """Run the prototype Python kernel locally.

    Parameters
    ----------
    input_path : str
        Path to kernel_input.json.
    output_path : str
        Path for kernel_output.json.
    precision_bits : int
        Precision bits for interval arithmetic.
    timeout : int
        Execution timeout in seconds.

    Returns
    -------
    Dict[str, Any]
        Parsed kernel output JSON.
    """
    # Find prototype kernel script
    if KERNEL_LOCAL_PY and os.path.exists(KERNEL_LOCAL_PY):
        kernel_script = KERNEL_LOCAL_PY
    else:
        # Default location in repo
        repo_root = Path(__file__).resolve().parent.parent
        kernel_script = str(repo_root / "kernel" / "prototype" / "prototype_kernel.py")

    if not os.path.exists(kernel_script):
        raise KernelClientError(
            f"Prototype kernel not found at {kernel_script}. "
            "Set ESM_KERNEL_LOCAL_PY or install the kernel module."
        )

    cmd = [
        "python", kernel_script,
        input_path,
        output_path,
        "--precision", str(precision_bits),
    ]

    logger.info(f"Running prototype kernel: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Prototype kernel failed: {result.stderr}")
            raise KernelClientError(
                f"Prototype kernel failed with exit code {result.returncode}: {result.stderr}"
            )

        # Parse output
        if not os.path.exists(output_path):
            raise KernelClientError(f"Kernel output file not created: {output_path}")

        with open(output_path, 'r') as f:
            output = json.load(f)

        logger.info("Prototype kernel completed successfully")
        return output

    except subprocess.TimeoutExpired:
        raise KernelClientError(f"Prototype kernel timed out after {timeout}s")
    except json.JSONDecodeError as e:
        raise KernelClientError(f"Invalid kernel output JSON: {e}")


def _run_docker_kernel(
    input_path: str,
    output_path: str,
    precision_bits: int,
    mode: str,
    timeout: int,
) -> Dict[str, Any]:
    """Run the verified kernel in Docker container.

    Parameters
    ----------
    input_path : str
        Path to kernel_input.json.
    output_path : str
        Path for kernel_output.json.
    precision_bits : int
        Precision bits for interval arithmetic.
    mode : str
        Kernel mode: "arb" or "mpfi".
    timeout : int
        Execution timeout in seconds.

    Returns
    -------
    Dict[str, Any]
        Parsed kernel output JSON.
    """
    # Check Docker is available
    if not shutil.which("docker"):
        raise KernelClientError(
            "Docker not found. Install Docker or use mode='prototype' for local testing."
        )

    # Prepare output directory
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Create empty output file so Docker can mount it
    with open(output_path, 'w') as f:
        json.dump({}, f)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_path}:/data/kernel_input.json:ro",
        "-v", f"{output_path}:/data/kernel_output.json:rw",
        "-e", f"PRECISION_BITS={precision_bits}",
        "-e", f"KERNEL_MODE={mode}",
        KERNEL_IMAGE,
        "/bin/kernel_binary",
        "/data/kernel_input.json",
        "/data/kernel_output.json",
    ]

    logger.info(f"Running Docker kernel: {KERNEL_IMAGE}")
    logger.debug(f"Docker command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Docker kernel failed: {result.stderr}")
            raise KernelClientError(
                f"Docker kernel failed with exit code {result.returncode}: {result.stderr}"
            )

        # Parse output
        with open(output_path, 'r') as f:
            output = json.load(f)

        logger.info("Docker kernel completed successfully")
        return output

    except subprocess.TimeoutExpired:
        raise KernelClientError(f"Docker kernel timed out after {timeout}s")
    except json.JSONDecodeError as e:
        raise KernelClientError(f"Invalid kernel output JSON: {e}")


def verify_kernel_output(output: Dict[str, Any]) -> bool:
    """Verify kernel output passes all checks.

    Parameters
    ----------
    output : Dict[str, Any]
        Parsed kernel output JSON.

    Returns
    -------
    bool
        True if all checks pass, False otherwise.
    """
    checks = output.get("checks", {})

    # Check theoretical bound
    if "theoretical_bound" in checks:
        if not checks["theoretical_bound"].get("pass", False):
            logger.warning("Theoretical bound check failed")
            return False

    # Check Wedin bound
    if "wedin_bound" in checks:
        if not checks["wedin_bound"].get("pass_estimate", False):
            logger.warning("Wedin bound check failed")
            return False

    return True


def run_kernel_and_verify(
    kernel_input_path: str,
    output_path: Optional[str] = None,
    precision_bits: int = 128,
    mode: str = "prototype",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run kernel and verify output passes all checks.

    Convenience function that runs the kernel and raises if checks fail.

    Parameters
    ----------
    kernel_input_path : str
        Path to kernel_input.json.
    output_path : Optional[str]
        Path for kernel_output.json.
    precision_bits : int
        Precision bits for interval arithmetic.
    mode : str
        Kernel mode: "prototype", "arb", or "mpfi".
    timeout : Optional[int]
        Execution timeout in seconds.

    Returns
    -------
    Dict[str, Any]
        Parsed kernel output JSON.

    Raises
    ------
    KernelClientError
        If kernel execution fails or checks fail.
    """
    output = run_kernel(kernel_input_path, output_path, precision_bits, mode, timeout)

    if not verify_kernel_output(output):
        raise KernelClientError("Kernel output verification failed")

    return output


def get_kernel_diagnostics() -> Dict[str, Any]:
    """Get diagnostic information about kernel availability.

    Returns
    -------
    Dict[str, Any]
        Diagnostic information including Docker status, image, local kernel path, etc.
    """
    diagnostics = {
        "kernel_image": KERNEL_IMAGE,
        "kernel_local_py": KERNEL_LOCAL_PY,
        "kernel_timeout": KERNEL_TIMEOUT,
        "docker_available": shutil.which("docker") is not None,
        "prototype_available": False,
    }

    # Check prototype kernel availability
    if KERNEL_LOCAL_PY and os.path.exists(KERNEL_LOCAL_PY):
        diagnostics["prototype_available"] = True
        diagnostics["prototype_path"] = KERNEL_LOCAL_PY
    else:
        repo_root = Path(__file__).resolve().parent.parent
        prototype_path = repo_root / "kernel" / "prototype" / "prototype_kernel.py"
        if prototype_path.exists():
            diagnostics["prototype_available"] = True
            diagnostics["prototype_path"] = str(prototype_path)

    # Check Docker image if Docker is available
    if diagnostics["docker_available"]:
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", KERNEL_IMAGE],
                capture_output=True,
                timeout=10,
            )
            diagnostics["docker_image_available"] = result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            diagnostics["docker_image_available"] = False
    else:
        diagnostics["docker_image_available"] = False

    return diagnostics


__all__ = [
    "run_kernel",
    "run_kernel_and_verify",
    "verify_kernel_output",
    "get_kernel_diagnostics",
    "KernelClientError",
    "KERNEL_IMAGE",
]
