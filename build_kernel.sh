#!/usr/bin/env bash
# Robust build_kernel.sh for UELAT kernel_verified.so
#
# Usage:
#   ./build_kernel.sh            # build kernel
#   ./build_kernel.sh clean      # remove build artifacts
#
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Allow user/CI to override the UELAT directory or to provide a prebuilt kernel
# VERIFIED_KERNEL_PATH (absolute or relative) will be used if set and exists.
: "${VERIFIED_KERNEL_PATH:=}"

# Initialize UELAT_DIR from environment if provided (do not clobber an export)
UELAT_DIR="${UELAT_DIR:-}"

# Multi-strategy discovery for UELAT_DIR:
# 1) If env UELAT_DIR is provided and points to a directory, use it.
# 2) If $ROOT_DIR/UELAT exists, use that.
# 3) If VERIFIED_KERNEL_PATH points to a .so, prefer using that (skip source build).
# 4) Otherwise, try to find a UELAT directory in repo with 'find'.
# Strategy 1: explicitly provided by environment
if [ -n "${UELAT_DIR:-}" ]; then
  if [ -d "${UELAT_DIR}" ]; then
    echo "[kernel] Using UELAT_DIR from environment: ${UELAT_DIR}"
  else
    echo "[kernel] WARNING: UELAT_DIR environment variable set but path does not exist: ${UELAT_DIR}"
    UELAT_DIR=""
  fi
fi

# Strategy 2: conventional path
if [ -z "$UELAT_DIR" ] && [ -d "${ROOT_DIR}/UELAT" ]; then
  UELAT_DIR="${ROOT_DIR}/UELAT"
fi

# Strategy 3: if a prebuilt kernel is supplied, prefer it (skip building sources)
if [ -n "${VERIFIED_KERNEL_PATH:-}" ] && [ -f "${VERIFIED_KERNEL_PATH}" ]; then
  echo "[kernel] VERIFIED_KERNEL_PATH provided and file exists: ${VERIFIED_KERNEL_PATH}"
  echo "[kernel] Using prebuilt kernel; skipping UELAT source build."
  UELAT_DIR=""
fi

# Strategy 4: search for candidate UELAT directories
if [ -z "$UELAT_DIR" ]; then
  echo "[kernel] UELAT not found at ${ROOT_DIR}/UELAT; searching repository for UELAT directory..."
  # search up to depth 2 to avoid long scans; this returns the first plausible match
  candidate="$(find "${ROOT_DIR}" -maxdepth 3 -type d -name UELAT -print -quit || true)"
  if [ -n "$candidate" ]; then
    echo "[kernel] Found candidate UELAT at: $candidate"
    UELAT_DIR="$candidate"
  fi
fi

# Final check: if no UELAT_DIR but a verified kernel path exists, use the kernel artifact path.
if [ -z "${UELAT_DIR:-}" ] && [ -n "${VERIFIED_KERNEL_PATH:-}" ] && [ -f "${VERIFIED_KERNEL_PATH}" ]; then
  echo "[kernel] No UELAT source directory, but VERIFIED_KERNEL_PATH exists; using artifact at ${VERIFIED_KERNEL_PATH}"
  # set KERNEL_OUTPUT based on provided artifact
  KERNEL_OUTPUT="${VERIFIED_KERNEL_PATH}"
  BUILD_FROM_SOURCES=0
else
  BUILD_FROM_SOURCES=1
  if [ -n "${KERNEL_OUTPUT:-}" ]; then
    echo "[kernel] Using externally set KERNEL_OUTPUT=${KERNEL_OUTPUT}"
  else
    # set default values used later
    KERNEL_OUTPUT="${UELAT_DIR:-./UELAT}/kernel_verified.so"
  fi
fi

# If we intend to build from sources but UELAT_DIR is still missing, error with actionable guidance
if [ "$BUILD_FROM_SOURCES" -eq 1 ] && [ -z "${UELAT_DIR:-}" ]; then
  echo "[kernel] ERROR: UELAT directory not found; cannot build kernel from sources."
  echo "[kernel] Searched: ${ROOT_DIR}/UELAT and repo. If you expect sources in the repo, ensure they exist or set VERIFIED_KERNEL_PATH to a prebuilt kernel."
  echo "[kernel] CI guidance: ensure the checkout includes the UELAT directory (use 'actions/checkout' with submodules if relevant) or upload the built kernel as the 'verified-kernel' artifact."
  exit 1
fi

# For building, set BUILD_DIR relative to UELAT_DIR
if [ "$BUILD_FROM_SOURCES" -eq 1 ]; then
  BUILD_DIR="${UELAT_DIR}/_build"
  mkdir -p "$BUILD_DIR"
else
  # BUILD_DIR still useful for logs even when using artifact
  BUILD_DIR="${ROOT_DIR}/UELAT/_build"
  mkdir -p "$BUILD_DIR" || true
fi

# Tools (allow environment overrides)
COQC="${COQC:-coqc}"
OCAMLOPT="${OCAMLOPT:-ocamlopt}"
OCAMLFIND="${OCAMLFIND:-ocamlfind}"

# Preflight tool checks (only when building from sources)
require_tool() {
  local tool_name="$1"
  local tool_label="$2"
  if ! command -v "$tool_name" >/dev/null 2>&1; then
    echo "[kernel] ERROR: Required tool '${tool_label}' not found in PATH."
    echo "[kernel] Install the Coq/OCaml toolchain or set COQC/OCAMLOPT/OCAMLFIND to valid executables."
    exit 1
  fi
}

if [ "$BUILD_FROM_SOURCES" -eq 1 ]; then
  COQC_BIN="${COQC%% *}"
  OCAMLOPT_BIN="${OCAMLOPT%% *}"
  OCAMLFIND_BIN="${OCAMLFIND%% *}"
  require_tool "$COQC_BIN" "$COQC"
  require_tool "$OCAMLOPT_BIN" "$OCAMLOPT"
  require_tool "$OCAMLFIND_BIN" "$OCAMLFIND"
fi

# Diagnostic trap
print_diagnostics() {
  echo "=== Kernel build diagnostics ==="
  echo "ROOT_DIR: $ROOT_DIR"
  echo "UELAT_DIR: ${UELAT_DIR:-<empty>}"
  echo "BUILD_FROM_SOURCES: $BUILD_FROM_SOURCES"
  echo "KERNEL_OUTPUT: $KERNEL_OUTPUT"
  echo "--- versions ---"
  $COQC --version 2>&1 || echo "coqc not found"
  $OCAMLOPT -version 2>&1 || echo "ocamlopt not found"
  $OCAMLFIND --version 2>&1 || echo "ocamlfind not found"
  echo "--- listing ---"
  if [ -n "${UELAT_DIR:-}" ] && [ -d "$UELAT_DIR" ]; then
    ls -la "$UELAT_DIR" || true
  fi
  ls -la "$BUILD_DIR" || true
  echo "=== end diagnostics ==="
}

trap print_diagnostics ERR

# Handle clean mode safely: check $# to avoid unbound variable usage
if [ "${1:-}" = "clean" ] || [ "${1:-}" = "clean-all" ]; then
  echo "[kernel] Cleaning build artifacts..."
  if [ -n "$UELAT_DIR" ] && [ -d "$UELAT_DIR" ]; then
    rm -rf "${BUILD_DIR}" "${KERNEL_OUTPUT}"
    echo "[kernel] Clean complete."
    exit 0
  else
    echo "[kernel] No UELAT directory to clean."
    exit 0
  fi
fi

if [ "$BUILD_FROM_SOURCES" -eq 0 ]; then
  if [ ! -f "$KERNEL_OUTPUT" ]; then
    echo "[kernel] ERROR: VERIFIED_KERNEL_PATH was set but kernel artifact not found at ${KERNEL_OUTPUT}" >&2
    exit 1
  fi
  echo "[kernel] Using prebuilt kernel artifact at ${KERNEL_OUTPUT}"
  exit 0
fi

if [ -z "${UELAT_DIR:-}" ] || [ ! -d "${UELAT_DIR}" ]; then
  echo "[kernel] ERROR: UELAT directory not found at ${UELAT_DIR}" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

echo "[kernel] Using COQC=${COQC}, OCAMLOPT=${OCAMLOPT}, OCAMLFIND=${OCAMLFIND}"
$COQC --version 2>&1 || true
$OCAMLOPT -version 2>&1 || true

pushd "$UELAT_DIR" > /dev/null

echo "[kernel] Step 1: Verifying Coq proofs..."

# Recommended: compile in a deterministic order. Ensure spectral_bounds is compiled first.
# If your project has other dependencies, expand this list or use coq_makefile / _CoqProject.
VFILES=(spectral_bounds.v CertificateCore.v CertificateProofs.v Extraction.v)

# Ensure logs directory
mkdir -p "$BUILD_DIR"

for vf in "${VFILES[@]}"; do
  if [ -f "$vf" ]; then
    echo "[kernel] Compiling ${vf} ..."
    # Compile with logical mapping: -Q . "" maps the current folder to the empty module namespace.
    # Use tee to save logs for diagnostics.
    if ! ${COQC} -Q . "" "$vf" 2>&1 | tee "${BUILD_DIR}/coq_${vf}.log"; then
      echo "[kernel] ERROR: coqc failed on ${vf}. See ${BUILD_DIR}/coq_${vf}.log"
      popd > /dev/null
      exit 1
    fi
  else
    echo "[kernel] Note: ${vf} not present; skipping."
  fi
done

# After compilation, verify .vo files exist for the modules we need
for mod in spectral_bounds CertificateCore CertificateProofs Extraction; do
  if [ -f "${mod}.vo" ]; then
    echo "[kernel] Found ${mod}.vo"
  else
    echo "[kernel] WARNING: Expected ${mod}.vo not found."
  fi
done

# Extraction step: Extraction.v should produce kernel_verified.ml (via Coq extraction).
if [ -f "Extraction.v" ]; then
  echo "[kernel] Running extraction (Extraction.v) if kernel_verified.ml missing..."
  if [ ! -f "kernel_verified.ml" ]; then
  if ! ${COQC} -Q . "" Extraction.v 2>&1 | tee "${BUILD_DIR}/coq_Extraction.v.log"; then
      echo "[kernel] ERROR: Extraction failed. See ${BUILD_DIR}/coq_Extraction.v.log"
      popd > /dev/null
      exit 1
    fi
  else
    echo "[kernel] kernel_verified.ml already exists; skipping extraction."
  fi
else
  echo "[kernel] No Extraction.v found; skipping extraction step."
fi

if [ ! -f "kernel_verified.ml" ]; then
  echo "[kernel] ERROR: kernel_verified.ml not found after extraction. Cannot proceed."
  popd > /dev/null
  exit 1
fi

# Register OCaml functions with the callback mechanism so C code can find them via caml_named_value()
# Without this, caml_named_value("kernel_api_certificate") returns NULL and causes a segfault
echo "[kernel] Appending Callback.register to kernel_verified.ml..."
cat >> kernel_verified.ml <<'CALLBACK_REGISTER'

(* Register functions for C interop via caml_named_value *)
let () = Callback.register "kernel_api_certificate" kernel_api_certificate
let () = Callback.register "kernel_api_residual" kernel_api_residual
let () = Callback.register "kernel_api_bound" kernel_api_bound
let () = Callback.register "kernel_api_frobenius_norm" kernel_api_frobenius_norm
CALLBACK_REGISTER

echo "[kernel] Step 2: Extracted kernel_verified.ml. Proceeding to OCaml compile..."

# Prepare build dir and compile OCaml file to cmx
mkdir -p "$BUILD_DIR"
echo "[kernel] Compiling kernel_verified.* into ${BUILD_DIR}/ (ensuring .cmi exists)"

if [ -f "kernel_verified.mli" ]; then
  echo "[kernel] Found kernel_verified.mli; compiling to .cmi"
  if ! ${OCAMLOPT} -c -I +unix kernel_verified.mli; then
    echo "[kernel] ERROR: ocamlopt failed to compile kernel_verified.mli"
    popd > /dev/null
    exit 1
  fi
  mv -f kernel_verified.cmi "${BUILD_DIR}/" || true
fi

echo "[kernel] Compiling kernel_verified.ml into ${BUILD_DIR}/kernel_verified.cmx"
if ! ${OCAMLOPT} -c -I +unix -I "${BUILD_DIR}" kernel_verified.ml -o "${BUILD_DIR}/kernel_verified.cmx"; then
  echo "[kernel] ERROR: ocamlopt failed to compile kernel_verified.ml"
  popd > /dev/null
  exit 1
fi

# Create the C wrapper (same as your previous wrapper; keep behaviour identical).
cat > "${BUILD_DIR}/kernel_stub.c" <<'WRAPPER_END'
/* C wrapper for OCaml kernel functions
   Provides ctypes-compatible interfaces for Python to call verified kernel
*/

#include <caml/mlvalues.h>
#include <caml/callback.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/memory.h>
#include <caml/fail.h>
#include <stdio.h>
#include <math.h>

/* Global OCaml runtime state */
static int ocaml_initialized = 0;

/* Initialize OCaml runtime once */
void kernel_init(void) {
    if (!ocaml_initialized) {
        char* argv[] = { "kernel", NULL };
        caml_startup(argv);
        ocaml_initialized = 1;
    }
}

/* Convert C double array to OCaml float list *)
   Matrix represented as list of lists (rows) *)
*/
value c_array_to_ocaml_matrix(double* data, int rows, int cols) {
    CAMLparam0();
    CAMLlocal2(matrix_list, row_list);
    int i, j;

    matrix_list = caml_alloc(rows, 0);  /* Allocate outer list (rows) */

    for (i = 0; i < rows; i++) {
        row_list = caml_alloc(cols, 0);  /* Allocate row list */

        for (j = 0; j < cols; j++) {
            caml_initialize(&Field(row_list, j),
                           caml_copy_double(data[i * cols + j]));
        }

        caml_initialize(&Field(matrix_list, i), row_list);
    }

    CAMLreturn(matrix_list);
}

/* Convert OCaml list to C double array */
void ocaml_list_to_c_array(value ocaml_list, double* out_array) {
    CAMLparam1(ocaml_list);

    *out_array = Double_val(ocaml_list);

    CAMLreturn0();
}

/* Main kernel wrapper: compute residual and bound

   Signature: (double array, int, int) -> (double array, int, int) ->
              (double array, int, int) -> double -> double -> double ->
              (double, double)
*/
void kernel_compute_certificate_wrapper(
    double* X0_data, int X0_rows, int X0_cols,
    double* X1_data, int X1_rows, int X1_cols,
    double* A_data, int A_rows, int A_cols,
    double tail_energy,
    double semantic_divergence,
    double lipschitz_margin,
    double* out_residual,
    double* out_bound
) {
    CAMLparam0();
    CAMLlocal5(ocaml_X0, ocaml_X1, ocaml_A, result, kernel_func);

    kernel_init();

    /* Convert C arrays to OCaml matrices (lists of lists) */
    ocaml_X0 = c_array_to_ocaml_matrix(X0_data, X0_rows, X0_cols);
    ocaml_X1 = c_array_to_ocaml_matrix(X1_data, X1_rows, X1_cols);
    ocaml_A = c_array_to_ocaml_matrix(A_data, A_rows, A_cols);

    /* Get reference to OCaml kernel_api_certificate function */
    kernel_func = caml_named_value("kernel_api_certificate");
    if (kernel_func == NULL) {
        caml_failwith("kernel_api_certificate not found in OCaml");
    }

    /* Call: kernel_compute_certificate(X0, X1, A, te, sd, lm) -> (residual, bound) */
    result = caml_callbackN(kernel_func, 6,
        (value[]){
            ocaml_X0,
            ocaml_X1,
            ocaml_A,
            caml_copy_double(tail_energy),
            caml_copy_double(semantic_divergence),
            caml_copy_double(lipschitz_margin)
        });

    /* Extract (residual, bound) tuple from result */
    *out_residual = Double_val(Field(result, 0));
    *out_bound = Double_val(Field(result, 1));

    CAMLreturn0();
}

/* Alternate entry points for modular calling */
void kernel_compute_residual_wrapper(
    double* X0_data, int X0_rows, int X0_cols,
    double* X1_data, int X1_rows, int X1_cols,
    double* A_data, int A_rows, int A_cols,
    double* out_residual
) {
    CAMLparam0();
    CAMLlocal4(ocaml_X0, ocaml_X1, ocaml_A, result);

    kernel_init();

    ocaml_X0 = c_array_to_ocaml_matrix(X0_data, X0_rows, X0_cols);
    ocaml_X1 = c_array_to_ocaml_matrix(X1_data, X1_rows, X1_cols);
    ocaml_A = c_array_to_ocaml_matrix(A_data, A_rows, A_cols);

    /* Note: kernel_api_residual not exported yet, using full certificate */
    /* In production, export individual functions from CertificateCore.v */

    CAMLreturn0();
}
WRAPPER_END

echo "[kernel] Created kernel_stub.c in ${BUILD_DIR}"

# Step 3: Create object file with embedded OCaml runtime
echo "[kernel] Creating object file with OCaml runtime..."
if ! ${OCAMLOPT} -output-obj -I "${BUILD_DIR}" -o "${BUILD_DIR}/kernel_verified.o" "${BUILD_DIR}/kernel_verified.cmx"; then
  echo "[kernel] ERROR: Failed to create object file with OCaml runtime"
  popd > /dev/null
  exit 1
fi

# Step 4: Compile C stub
echo "[kernel] Compiling C stub..."
OCAML_WHERE=$(${OCAMLOPT} -where)
if ! gcc -c -fPIC -I "${OCAML_WHERE}" "${BUILD_DIR}/kernel_stub.c" -o "${BUILD_DIR}/kernel_stub.o"; then
  echo "[kernel] ERROR: Failed to compile C stub"
  popd > /dev/null
  exit 1
fi

# Step 5: Link everything into shared library
echo "[kernel] Linking to shared library..."
# Find the OCaml runtime library (libasmrun_pic.a for PIC code, or libasmrun.a)
ASMRUN_LIB=""
for lib in "${OCAML_WHERE}/libasmrun_pic.a" "${OCAML_WHERE}/libasmrun.a"; do
  if [ -f "$lib" ]; then
    ASMRUN_LIB="$lib"
    break
  fi
done

if [ -z "$ASMRUN_LIB" ]; then
  echo "[kernel] WARNING: Could not find libasmrun, trying default linker search"
  ASMRUN_LIB="-lasmrun"
fi

echo "[kernel] Using OCaml runtime: ${ASMRUN_LIB}"

if ! gcc -shared -fPIC -o "${KERNEL_OUTPUT}" \
    "${BUILD_DIR}/kernel_verified.o" \
    "${BUILD_DIR}/kernel_stub.o" \
    "${ASMRUN_LIB}" \
    -L"${OCAML_WHERE}" \
    -lm -ldl -lpthread; then
  echo "[kernel] ERROR: Failed to create shared library"
  popd > /dev/null
  exit 1
fi

if [ -f "${KERNEL_OUTPUT}" ]; then
  echo "[kernel] SUCCESS: Kernel built at ${KERNEL_OUTPUT}"
  ls -lh "${KERNEL_OUTPUT}"
  popd > /dev/null
  exit 0
else
  echo "[kernel] ERROR: Kernel build did not produce ${KERNEL_OUTPUT}"
  popd > /dev/null
  exit 1
fi
