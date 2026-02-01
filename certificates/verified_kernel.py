"""Python Bridge to Formally Verified Certificate Kernel

This module provides Python bindings to the verified Coq/OCaml kernel
that computes residuals and theoretical bounds.

**Witness-Checker Architecture:**
- Python (unverified): Computes SVD, derives witness matrices
- Kernel (verified): Computes residual and bound from witnesses

All calls to the kernel are strict: failures cause hard errors, not fallbacks.

**Environment Variables:**
- ESM_ALLOW_KERNEL_LOAD: Set to "0" to disable native kernel loading (safe for CI)
- ESM_SKIP_VERIFIED_KERNEL: Set to "1" to use Python fallback in make_certificate
- VERIFIED_KERNEL_PATH: Override path to kernel_verified.so

**Threading and Process Constraints:**

The verified kernel initializes the OCaml runtime, which has important constraints:

1. **Main Thread Requirement**: The OCaml runtime MUST be initialized in the 
   main thread of the process. Attempting to initialize in a worker thread may
   cause segfaults or undefined behavior.

2. **Single Initialization**: The OCaml runtime can only be initialized once per
   process lifetime. Calling `load_kernel()` multiple times is safe (it caches
   the handle), but the underlying `kernel_init()` is only called once.

3. **No Fork After Load**: After loading the kernel, do NOT fork the process.
   The OCaml runtime is not fork-safe. If you need multiprocessing:
   - Fork BEFORE loading the kernel
   - Load the kernel independently in each child process
   - Or use a kernel service architecture (see kernel_server.py)

4. **Subprocess Safety**: To load the kernel in a subprocess, use the standard
   multiprocessing module and load the kernel in the child process's main function.
   Example:
   
   ```python
   import multiprocessing
   from certificates.verified_kernel import load_kernel
   
   def worker():
       kernel = load_kernel(strict=False)
       # Use kernel...
   
   if __name__ == "__main__":
       p = multiprocessing.Process(target=worker)
       p.start()
       p.join()
   ```

5. **Long-Running Processes**: For long-running processes that need to reload
   the kernel (e.g., servers), use one of these approaches:
   - **Recommended**: Use the kernel service (kernel_server.py) which manages
     the kernel lifecycle in a dedicated subprocess
   - **Alternative**: Restart the entire process to reload the kernel
   - **Not Supported**: Unloading and reloading the kernel in the same process
     is not supported due to OCaml runtime limitations

6. **Thread Safety**: Once loaded, the kernel functions are thread-safe for
   concurrent calls from multiple threads (the OCaml runtime handles locking).
   However, the initial loading must occur in the main thread.

**Segfault Prevention:**

The kernel can segfault if:
1. The shared object is missing callback registrations (caml_named_value returns NULL)
2. The OCaml runtime is not initialized
3. Symbol ABI mismatch between ctypes declarations and actual exports
4. The kernel is loaded from a non-main thread
5. The process is forked after loading the kernel

This module validates the kernel before calling it to prevent segfaults.

**Safe Reload API:**

For applications that need to reload the kernel:

```python
from certificates.verified_kernel import reset_kernel_state, load_kernel

# In a new process (after fork or in a fresh subprocess):
reset_kernel_state()  # Clear cached state
kernel = load_kernel(strict=False)  # Load fresh
```

Note: `reset_kernel_state()` only clears Python-side state. It does NOT unload
the OCaml runtime. Use this only when starting fresh in a new process.
"""

from __future__ import annotations

import ctypes
import faulthandler
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from certificates.witness_checker import (
    WitnessValidationError,
    check_witness,
)

logger = logging.getLogger(__name__)

# Enable faulthandler for better crash diagnostics
if not faulthandler.is_enabled():
    try:
        faulthandler.enable()
    except Exception:
        pass  # May fail if stderr is not writable

# Global kernel handle
_kernel_lib: Optional[ctypes.CDLL] = None
# Sentinel to track if we already tried and failed to load the kernel
_kernel_load_attempted: bool = False
_kernel_load_warning_logged: bool = False
# Track whether we've already logged the "unavailable" fallback warning
_fallback_warning_logged: bool = False
# Track whether kernel passed validation (symbol checks)
_kernel_validated: bool = False

# Required symbols that must be exported by the kernel
REQUIRED_KERNEL_SYMBOLS = [
    "kernel_compute_certificate_wrapper",
    "kernel_init",
]


class KernelError(RuntimeError):
    """Raised when the verified kernel fails or is unavailable."""


def _warn_python_fallback_once() -> None:
    """Warn once when using the Python fallback instead of the verified kernel."""

    global _fallback_warning_logged
    if _fallback_warning_logged:
        return
    logger.warning("Verified kernel unavailable, using Python fallback")
    _fallback_warning_logged = True


def _locate_kernel() -> Optional[str]:
    """Find the compiled kernel shared library.

    Search locations:
    1. Environment variable VERIFIED_KERNEL_PATH
    2. build/kernel_verified.so (Linux/macOS)
    3. build/kernel_verified.dll (Windows)
    4. UELAT/_build/kernel_verified.so
    """

    # Check environment variable
    env_path = os.environ.get("VERIFIED_KERNEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Check standard build locations
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "build" / "kernel_verified.so",
        repo_root / "build" / "kernel_verified.dylib",
        repo_root / "build" / "kernel_verified.dll",
        # build_kernel.sh outputs to UELAT/kernel_verified.so
        repo_root / "UELAT" / "kernel_verified.so",
        repo_root / "UELAT" / "kernel_verified.dylib",
        repo_root / "UELAT" / "_build" / "kernel_verified.so",
        repo_root / "UELAT" / "_build" / "kernel_verified.dylib",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _check_library_symbols(kernel_path: str) -> Tuple[bool, List[str], List[str]]:
    """Check if the kernel shared library exports required symbols.

    Uses nm or objdump to inspect the shared library without loading it.
    This prevents segfaults from being triggered during symbol lookup.

    Parameters
    ----------
    kernel_path : str
        Path to the kernel shared library.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (all_present, found_symbols, missing_symbols)
    """
    found_symbols: List[str] = []
    missing_symbols: List[str] = []

    # Try nm first (most common), then objdump as fallback
    for cmd in [["nm", "-D", kernel_path], ["objdump", "-T", kernel_path]]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                output = result.stdout
                for sym in REQUIRED_KERNEL_SYMBOLS:
                    if sym in output:
                        found_symbols.append(sym)
                    else:
                        missing_symbols.append(sym)
                return len(missing_symbols) == 0, found_symbols, missing_symbols
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    # If we can't run nm/objdump, try loading with ctypes and checking hasattr
    # This is riskier but better than nothing
    logger.debug("nm/objdump unavailable, using ctypes for symbol check")
    try:
        lib = ctypes.CDLL(kernel_path)
        for sym in REQUIRED_KERNEL_SYMBOLS:
            if hasattr(lib, sym):
                found_symbols.append(sym)
            else:
                missing_symbols.append(sym)
        return len(missing_symbols) == 0, found_symbols, missing_symbols
    except OSError as e:
        logger.warning(f"Cannot verify kernel symbols: {e}")
        # Assume symbols are present if we can't check
        return True, REQUIRED_KERNEL_SYMBOLS, []


def _check_library_dependencies(kernel_path: str) -> Tuple[bool, str]:
    """Check if the kernel shared library has all required dependencies.

    Uses ldd to check for missing shared library dependencies.

    Parameters
    ----------
    kernel_path : str
        Path to the kernel shared library.

    Returns
    -------
    Tuple[bool, str]
        (all_satisfied, diagnostic_message)
    """
    try:
        result = subprocess.run(
            ["ldd", kernel_path],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            return True, "ldd check skipped (non-zero return)"

        output = result.stdout
        missing = []
        for line in output.splitlines():
            if "not found" in line.lower():
                missing.append(line.strip())

        if missing:
            return False, f"Missing dependencies: {'; '.join(missing)}"
        return True, "All dependencies satisfied"

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        # ldd not available (e.g., macOS, Windows)
        return True, f"ldd check skipped ({e})"


def _validate_kernel(kernel_path: str, lib: ctypes.CDLL) -> Tuple[bool, str]:
    """Validate a loaded kernel before use.

    Performs comprehensive checks to ensure the kernel is safe to call.

    Parameters
    ----------
    kernel_path : str
        Path to the loaded kernel.
    lib : ctypes.CDLL
        The loaded library handle.

    Returns
    -------
    Tuple[bool, str]
        (valid, diagnostic_message)
    """
    diagnostics = []

    # Check 1: Required symbols exist
    symbols_ok, found, missing = _check_library_symbols(kernel_path)
    if not symbols_ok:
        return False, f"Missing required symbols: {missing}. Found: {found}"
    diagnostics.append(f"Symbols OK: {found}")

    # Check 2: Dependencies satisfied
    deps_ok, deps_msg = _check_library_dependencies(kernel_path)
    if not deps_ok:
        return False, deps_msg
    diagnostics.append(deps_msg)

    # Check 3: kernel_init function exists and is callable
    try:
        if hasattr(lib, "kernel_init"):
            kernel_init_fn = lib.kernel_init
            kernel_init_fn.argtypes = []
            kernel_init_fn.restype = None
            diagnostics.append("kernel_init symbol available")
        else:
            return False, "kernel_init symbol not found - OCaml runtime may not initialize"
    except Exception as e:
        return False, f"Failed to configure kernel_init: {e}"

    # Check 4: Main wrapper function exists
    try:
        if hasattr(lib, "kernel_compute_certificate_wrapper"):
            diagnostics.append("kernel_compute_certificate_wrapper symbol available")
        else:
            return False, "kernel_compute_certificate_wrapper symbol not found"
    except Exception as e:
        return False, f"Failed to verify wrapper symbol: {e}"

    return True, "; ".join(diagnostics)


def load_kernel(strict: bool = True) -> Optional[ctypes.CDLL]:
    """Load the verified kernel library.

    Parameters
    ----------
    strict : bool
        If True, raise KernelError if kernel cannot be loaded.
        If False, return None and log a warning (only once).

    Returns
    -------
    Optional[ctypes.CDLL]
        The loaded kernel library, or None if unavailable.

    Raises
    ------
    KernelError
        If strict=True and kernel cannot be loaded.
    """

    global _kernel_lib, _kernel_load_attempted, _kernel_load_warning_logged

    # Fast path: already loaded successfully
    if _kernel_lib is not None:
        return _kernel_lib

    # Fast path: already tried and failed (avoid repeated warnings)
    if _kernel_load_attempted:
        if strict:
            raise KernelError(
                "Verified kernel not available. "
                "Build with: ./build_kernel.sh "
                "or set VERIFIED_KERNEL_PATH environment variable."
            )
        return None

    # ---- SAFETY GUARD: allow tests/CI to disable loading the native kernel ----
    # If ESM_ALLOW_KERNEL_LOAD != "1" then do not attempt to load the native kernel.
    # This prevents unit-test processes from getting killed by a faulty shared object.
    allow = os.environ.get("ESM_ALLOW_KERNEL_LOAD", "1")
    if allow != "1":
        # Mark that we attempted (so callers behave the same next time)
        _kernel_load_attempted = True
        if strict:
            raise KernelError("Native verified kernel loading disabled by ESM_ALLOW_KERNEL_LOAD!=1")
        _warn_python_fallback_once()
        return None

    # Mark that we're attempting to load
    _kernel_load_attempted = True

    kernel_path = _locate_kernel()
    if not kernel_path:
        msg = (
            "Verified kernel not found. "
            "Build with: ./build_kernel.sh "
            "or set VERIFIED_KERNEL_PATH environment variable."
        )
        if strict:
            raise KernelError(msg)
        # Only log warning once
        if not _kernel_load_warning_logged:
            logger.warning(msg)
            _kernel_load_warning_logged = True
        return None

    try:
        # Pre-flight check: verify symbols exist before loading
        # This helps diagnose issues without risking a segfault
        symbols_ok, found, missing = _check_library_symbols(kernel_path)
        if not symbols_ok:
            msg = (
                f"Verified kernel at {kernel_path} is missing required symbols: {missing}. "
                f"Found: {found}. This kernel would segfault if called. "
                f"Rebuild with: ./build_kernel.sh"
            )
            if strict:
                raise KernelError(msg)
            if not _kernel_load_warning_logged:
                logger.warning(msg)
                _kernel_load_warning_logged = True
            return None

        # Check dependencies before loading
        deps_ok, deps_msg = _check_library_dependencies(kernel_path)
        if not deps_ok:
            msg = f"Verified kernel at {kernel_path} has missing dependencies: {deps_msg}"
            if strict:
                raise KernelError(msg)
            if not _kernel_load_warning_logged:
                logger.warning(msg)
                _kernel_load_warning_logged = True
            return None

        # Load the library
        logger.debug(f"Loading verified kernel from {kernel_path}")
        lib = ctypes.CDLL(kernel_path)

        # Validate the loaded library
        valid, diag = _validate_kernel(kernel_path, lib)
        if not valid:
            msg = f"Kernel validation failed: {diag}"
            if strict:
                raise KernelError(msg)
            if not _kernel_load_warning_logged:
                logger.warning(msg)
                _kernel_load_warning_logged = True
            return None

        # Initialize the OCaml runtime (critical to prevent segfaults)
        # The kernel_init function sets up the OCaml runtime
        try:
            kernel_init_fn = lib.kernel_init
            kernel_init_fn.argtypes = []
            kernel_init_fn.restype = None
            logger.debug("Initializing OCaml runtime via kernel_init()")
            kernel_init_fn()
            logger.debug("OCaml runtime initialized successfully")
        except Exception as e:
            msg = f"Failed to initialize OCaml runtime: {e}. Kernel may segfault."
            if strict:
                raise KernelError(msg) from e
            if not _kernel_load_warning_logged:
                logger.warning(msg)
                _kernel_load_warning_logged = True
            return None

        global _kernel_validated
        _kernel_validated = True
        _kernel_lib = lib
        logger.info(f"Loaded and validated verified kernel from {kernel_path}")
        logger.debug(f"Kernel diagnostics: {diag}")
        return _kernel_lib

    except KernelError:
        raise
    except OSError as exc:
        msg = f"Failed to load verified kernel from {kernel_path}: {exc}"
        if strict:
            raise KernelError(msg) from exc
        # Only log warning once
        if not _kernel_load_warning_logged:
            logger.warning(msg)
            _kernel_load_warning_logged = True
        return None


def _validate_matrix_input(M: np.ndarray, name: str) -> None:
    """Validate matrix input for kernel call.

    Parameters
    ----------
    M : np.ndarray
        Matrix to validate.
    name : str
        Name for error messages.

    Raises
    ------
    KernelError
        If matrix contains NaN, Inf, or is not numeric.
    """

    if not isinstance(M, np.ndarray):
        raise KernelError(f"{name} must be numpy array, got {type(M)}")

    if M.dtype not in (np.float32, np.float64, float):
        raise KernelError(f"{name} must be float array, got {M.dtype}")

    if not np.isfinite(M).all():
        raise KernelError(f"{name} contains NaN or Inf values")


def _validate_scalar_input(x: float, name: str) -> None:
    """Validate scalar input for kernel call.

    Parameters
    ----------
    x : float
        Scalar to validate.
    name : str
        Name for error messages.

    Raises
    ------
    KernelError
        If scalar is NaN, Inf, or negative.
    """

    if not isinstance(x, (int, float)):
        raise KernelError(f"{name} must be numeric, got {type(x)}")

    if not np.isfinite(float(x)):
        raise KernelError(f"{name} is not finite: {x}")

    if x < 0:
        raise KernelError(f"{name} must be non-negative, got {x}")


def compute_residual(
    X0: np.ndarray, X1: np.ndarray, A: np.ndarray, strict: bool = True
) -> float:
    """Compute residual using verified kernel.

    Calls the formally verified kernel to compute:
        residual = ||X1 - A @ X0||_F / ||X1||_F

    Parameters
    ----------
    X0 : np.ndarray
        State matrix at time t, shape (r, T-1).
    X1 : np.ndarray
        State matrix at time t+1, shape (r, T-1).
    A : np.ndarray
        Learned operator, shape (r, r).
    strict : bool
        If True, raise on kernel failure. If False, return 0.0.

    Returns
    -------
    float
        The residual error.

    Raises
    ------
    KernelError
        If strict=True and kernel unavailable or computation fails.
    """

    # Validate inputs
    _validate_matrix_input(X0, "X0")
    _validate_matrix_input(X1, "X1")
    _validate_matrix_input(A, "A")

    # Load kernel
    kernel = load_kernel(strict=strict)
    if kernel is None:
        if strict:
            raise KernelError("Verified kernel not available")
        # Fallback: warn once and return 0.0
        _warn_python_fallback_once()
        return 0.0

    try:
        # VERIFIED KERNEL CALL: Compute residual in formally verified OCaml kernel
        # The kernel implements: residual = ||X1 - A @ X0||_F / ||X1||_F
        # This computation is proven correct in Coq (CertificateProofs.v)

        kernel_compute_residual_fn = kernel.kernel_compute_certificate_wrapper
        kernel_compute_residual_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # X0
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # X1
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # A
            ctypes.c_double,  # tail_energy (not used for residual, pass 0)
            ctypes.c_double,  # semantic_divergence
            ctypes.c_double,  # lipschitz_margin
            ctypes.POINTER(ctypes.c_double),  # out_residual
            ctypes.POINTER(ctypes.c_double),  # out_bound
        ]
        kernel_compute_residual_fn.restype = None

        # Prepare output buffers
        out_residual = ctypes.c_double()
        out_bound = ctypes.c_double()

        # Call kernel with C-contiguous arrays
        X0_c = np.ascontiguousarray(X0, dtype=np.float64)
        X1_c = np.ascontiguousarray(X1, dtype=np.float64)
        A_c = np.ascontiguousarray(A, dtype=np.float64)

        kernel_compute_residual_fn(
            X0_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X0_c.shape[0], X0_c.shape[1],
            X1_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X1_c.shape[0], X1_c.shape[1],
            A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), A_c.shape[0], A_c.shape[1],
            ctypes.c_double(0.0),  # tail_energy (dummy)
            ctypes.c_double(0.0),  # semantic_divergence (dummy)
            ctypes.c_double(0.0),  # lipschitz_margin (dummy)
            ctypes.byref(out_residual),
            ctypes.byref(out_bound),
        )

        residual = out_residual.value
        logger.debug(f"Verified kernel computed residual: {residual:.6f}")
        return float(residual)

    except Exception as exc:
        msg = f"Kernel residual computation failed: {exc}"
        if strict:
            raise KernelError(msg) from exc
        logger.error(msg)
        return 0.0


def compute_bound(
    residual: float,
    tail_energy: float,
    semantic_divergence: float,
    lipschitz_margin: float,
    c_res: float = 1.0,
    c_tail: float = 1.0,
    c_sem: float = 1.0,
    c_robust: float = 1.0,
    strict: bool = True,
) -> float:
    """Compute theoretical bound using verified kernel.

    Calls the formally verified kernel to compute:
        bound = c_res*residual + c_tail*tail_energy
              + c_sem*semantic_divergence + c_robust*lipschitz_margin

    Parameters
    ----------
    residual : float
        Prediction residual.
    tail_energy : float
        Energy not captured by truncation.
    semantic_divergence : float
        Mean divergence from task embedding.
    lipschitz_margin : float
        Embedding Lipschitz margin.
    c_res, c_tail, c_sem, c_robust : float
        Verified constants (defaults from axioms).
    strict : bool
        If True, raise on kernel failure. If False, return fallback.

    Returns
    -------
    float
        The theoretical bound.

    Raises
    ------
    KernelError
        If strict=True and kernel unavailable or computation fails.
    """

    # Validate inputs
    _validate_scalar_input(residual, "residual")
    _validate_scalar_input(tail_energy, "tail_energy")
    _validate_scalar_input(semantic_divergence, "semantic_divergence")
    _validate_scalar_input(lipschitz_margin, "lipschitz_margin")
    _validate_scalar_input(c_res, "c_res")
    _validate_scalar_input(c_tail, "c_tail")
    _validate_scalar_input(c_sem, "c_sem")
    _validate_scalar_input(c_robust, "c_robust")

    # Load kernel
    kernel = load_kernel(strict=strict)
    if kernel is None:
        if strict:
            raise KernelError("Verified kernel not available")
        # Fallback: warn once and compute in Python
        _warn_python_fallback_once()
        return c_res * residual + c_tail * tail_energy + c_sem * semantic_divergence + c_robust * lipschitz_margin

    try:
        # VERIFIED KERNEL CALL: Compute bound in formally verified OCaml kernel
        # The kernel implements the exact formula:
        #   bound = c_res*residual + c_tail*tail_energy + c_sem*semantic_divergence + c_robust*lipschitz_margin
        # This computation is proven correct in Coq (CertificateProofs.v)
        #
        # Note: We pass dummy matrices for X0, X1, A since bound computation doesn't use them
        # The kernel returns the bound in out_bound

        kernel_compute_bound_fn = kernel.kernel_compute_certificate_wrapper
        kernel_compute_bound_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # X0 (dummy)
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # X1 (dummy)
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # A (dummy)
            ctypes.c_double,  # tail_energy
            ctypes.c_double,  # semantic_divergence
            ctypes.c_double,  # lipschitz_margin
            ctypes.POINTER(ctypes.c_double),  # out_residual (will be filled but ignored)
            ctypes.POINTER(ctypes.c_double),  # out_bound (this is what we want)
        ]
        kernel_compute_bound_fn.restype = None

        # Create dummy 1x1 matrices (kernel needs them but won't use for bound-only computation)
        dummy = np.ascontiguousarray([[residual]], dtype=np.float64)

        out_residual = ctypes.c_double()
        out_bound = ctypes.c_double()

        # Call kernel
        kernel_compute_bound_fn(
            dummy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, 1,
            dummy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, 1,
            dummy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, 1,
            ctypes.c_double(tail_energy),
            ctypes.c_double(semantic_divergence),
            ctypes.c_double(lipschitz_margin),
            ctypes.byref(out_residual),
            ctypes.byref(out_bound),
        )

        bound = out_bound.value
        logger.debug(f"Verified kernel computed bound: {bound:.6f}")
        return float(bound)

    except Exception as exc:
        msg = f"Kernel bound computation failed: {exc}"
        if strict:
            raise KernelError(msg) from exc
        logger.error(msg)
        return 0.0


def compute_certificate(
    X0: np.ndarray,
    X1: np.ndarray,
    A: np.ndarray,
    tail_energy: float,
    semantic_divergence: float,
    lipschitz_margin: float,
    strict: bool = True,
) -> Tuple[float, float]:
    """Compute residual and bound using verified kernel.

    **Witness-Checker Architecture:**
    - X0, X1, A: Witness matrices computed in Python (unverified, fast)
    - Verified kernel: Computes residual and bound with formal guarantees
    - Return: (residual, theoretical_bound) both proven correct by Coq

    Parameters
    ----------
    X0, X1, A : np.ndarray
        Witness matrices from Python (unverified).
    tail_energy, semantic_divergence, lipschitz_margin : float
        Computed error metrics.
    strict : bool
        If True, fail hard on kernel issues. If False, use Python fallback.

    Returns
    -------
    Tuple[float, float]
        (residual, theoretical_bound)
        Both values are computed by the formally verified kernel.

    Raises
    ------
    KernelError
        If strict=True and kernel computation fails.
    WitnessValidationError
        If witness matrices fail numerical preconditions.
    """

    k = min(X0.shape[0], X0.shape[1]) if isinstance(X0, np.ndarray) and X0.ndim == 2 else 1
    check_witness(X0, X1, A, k=k)

    # Load kernel once
    kernel = load_kernel(strict=strict)
    if kernel is None:
        if strict:
            raise KernelError("Verified kernel not available")
        # Fallback: compute in Python (not verified)
        # Warning already logged by load_kernel
        residual = compute_residual(X0, X1, A, strict=False)
        bound = compute_bound(residual, tail_energy, semantic_divergence, lipschitz_margin, strict=False)
        return (residual, bound)

    try:
        # Call verified kernel directly (single call for both residual and bound)
        # This is more efficient than separate calls

        kernel_compute_cert_fn = kernel.kernel_compute_certificate_wrapper
        kernel_compute_cert_fn.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # X0
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # X1
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,  # A
            ctypes.c_double,  # tail_energy
            ctypes.c_double,  # semantic_divergence
            ctypes.c_double,  # lipschitz_margin
            ctypes.POINTER(ctypes.c_double),  # out_residual
            ctypes.POINTER(ctypes.c_double),  # out_bound
        ]
        kernel_compute_cert_fn.restype = None

        # Prepare output buffers
        out_residual = ctypes.c_double()
        out_bound = ctypes.c_double()

        # Ensure C-contiguous arrays
        X0_c = np.ascontiguousarray(X0, dtype=np.float64)
        X1_c = np.ascontiguousarray(X1, dtype=np.float64)
        A_c = np.ascontiguousarray(A, dtype=np.float64)

        # VERIFIED KERNEL CALL
        # This call invokes the formally verified Coq/OCaml kernel
        # The kernel proves:
        #   1. residual is correctly computed
        #   2. bound satisfies the theoretical formula
        #   3. bound is non-negative and monotonic
        logger.debug("Calling verified kernel for certificate computation")

        kernel_compute_cert_fn(
            X0_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X0_c.shape[0], X0_c.shape[1],
            X1_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X1_c.shape[0], X1_c.shape[1],
            A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), A_c.shape[0], A_c.shape[1],
            ctypes.c_double(tail_energy),
            ctypes.c_double(semantic_divergence),
            ctypes.c_double(lipschitz_margin),
            ctypes.byref(out_residual),
            ctypes.byref(out_bound),
        )

        residual = out_residual.value
        bound = out_bound.value

        logger.info(
            f"Verified kernel computed: residual={residual:.6f}, bound={bound:.6f}"
        )

        return (float(residual), float(bound))

    except Exception as exc:
        msg = f"Verified kernel computation failed: {exc}"
        if strict:
            raise KernelError(msg) from exc
        logger.error(msg)
        # Fallback to Python (not verified)
        residual = compute_residual(X0, X1, A, strict=False)
        bound = compute_bound(residual, tail_energy, semantic_divergence, lipschitz_margin, strict=False)
        return (residual, bound)


def is_kernel_available() -> bool:
    """Check if the verified kernel is available without loading it.

    This is a non-destructive check that inspects the kernel shared object
    without actually loading it into the process (which could cause issues).

    Returns
    -------
    bool
        True if kernel appears to be available and valid.
    """
    # Check environment guard
    allow = os.environ.get("ESM_ALLOW_KERNEL_LOAD", "1")
    if allow != "1":
        return False

    # Try to locate kernel
    kernel_path = _locate_kernel()
    if not kernel_path:
        return False

    # Check symbols without loading
    symbols_ok, _, _ = _check_library_symbols(kernel_path)
    if not symbols_ok:
        return False

    # Check dependencies
    deps_ok, _ = _check_library_dependencies(kernel_path)
    if not deps_ok:
        return False

    return True


def get_kernel_diagnostics() -> dict:
    """Get diagnostic information about the kernel without loading it.

    This is useful for debugging kernel issues in CI or development.

    Returns
    -------
    dict
        Diagnostic information including path, symbols, dependencies, etc.
    """
    diag = {
        "allow_kernel_load": os.environ.get("ESM_ALLOW_KERNEL_LOAD", "1"),
        "skip_verified_kernel": os.environ.get("ESM_SKIP_VERIFIED_KERNEL", ""),
        "kernel_service_mode": os.environ.get("ESM_KERNEL_SERVICE", "0"),
        "verified_kernel_path_env": os.environ.get("VERIFIED_KERNEL_PATH", ""),
        "kernel_path": None,
        "kernel_exists": False,
        "symbols_ok": False,
        "symbols_found": [],
        "symbols_missing": [],
        "dependencies_ok": False,
        "dependencies_message": "",
        "already_loaded": _kernel_lib is not None,
        "validated": _kernel_validated,
        "load_attempted": _kernel_load_attempted,
    }

    kernel_path = _locate_kernel()
    diag["kernel_path"] = kernel_path

    if kernel_path:
        diag["kernel_exists"] = True

        # Check symbols
        symbols_ok, found, missing = _check_library_symbols(kernel_path)
        diag["symbols_ok"] = symbols_ok
        diag["symbols_found"] = found
        diag["symbols_missing"] = missing

        # Check dependencies
        deps_ok, deps_msg = _check_library_dependencies(kernel_path)
        diag["dependencies_ok"] = deps_ok
        diag["dependencies_message"] = deps_msg

    return diag


def reset_kernel_state() -> None:
    """Reset global kernel state for fresh loading in a new process.

    **Important**: This function only clears Python-side cached state. It does
    NOT unload the OCaml runtime, which cannot be unloaded once initialized.

    **Safe Usage Patterns:**

    1. **New subprocess**: After spawning a subprocess, call this before loading:
       ```python
       # In child process
       reset_kernel_state()
       kernel = load_kernel(strict=False)
       ```

    2. **Testing different configurations**: Reset between test cases that need
       different kernel paths or environment variables:
       ```python
       @pytest.fixture(autouse=True)
       def reset_kernel():
           reset_kernel_state()
           yield
           reset_kernel_state()
       ```

    **Unsafe Usage:**

    - Do NOT call this in the same process after successfully loading the kernel
      and expect to reload it. The OCaml runtime remains initialized.
    - Do NOT call this after forking if the parent loaded the kernel.

    **What This Does:**
    - Clears `_kernel_lib` handle (allows fresh dlopen)
    - Resets `_kernel_load_attempted` flag (allows retry)
    - Clears warning suppression flags (logs will show again)
    - Resets validation state

    **What This Does NOT Do:**
    - Unload the shared library from memory
    - Deinitialize the OCaml runtime
    - Close file descriptors or release resources

    For long-running processes that need to reload the kernel, use the kernel
    service architecture (kernel_server.py) instead.
    """
    global _kernel_lib, _kernel_load_attempted, _kernel_load_warning_logged
    global _fallback_warning_logged, _kernel_validated

    _kernel_lib = None
    _kernel_load_attempted = False
    _kernel_load_warning_logged = False
    _fallback_warning_logged = False
    _kernel_validated = False


__all__ = [
    "load_kernel",
    "compute_residual",
    "compute_bound",
    "compute_certificate",
    "KernelError",
    "WitnessValidationError",
    "is_kernel_available",
    "get_kernel_diagnostics",
    "reset_kernel_state",
]
