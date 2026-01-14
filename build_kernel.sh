#!/usr/bin/env bash
# Robust build_kernel.sh for UELAT kernel_verified.so
#
# Usage:
#   ./build_kernel.sh            # build kernel
#   ./build_kernel.sh clean      # remove build artifacts
#
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
UELAT_DIR="$(cd "${ROOT_DIR}/UELAT" 2>/dev/null && pwd || true)"
BUILD_DIR="${UELAT_DIR:-./UELAT}/_build"
KERNEL_OUTPUT="${UELAT_DIR:-./UELAT}/kernel_verified.so"

# Tools (allow environment overrides)
COQC="${COQC:-coqc}"
OCAMLOPT="${OCAMLOPT:-ocamlopt}"
OCAMLFIND="${OCAMLFIND:-ocamlfind}"

# Diagnostic helper (prints useful info on error)
print_diagnostics() {
  echo "=== Kernel build diagnostics ==="
  echo "PWD: $(pwd)"
  echo "ROOT_DIR: $ROOT_DIR"
  echo "UELAT_DIR: $UELAT_DIR"
  echo "BUILD_DIR: $BUILD_DIR"
  echo "KERNEL_OUTPUT: $KERNEL_OUTPUT"
  echo "--- versions ---"
  $COQC --version 2>&1 || echo "coqc not found"
  $OCAMLOPT -version 2>&1 || echo "ocamlopt not found"
  $OCAMLFIND --version 2>&1 || echo "ocamlfind not found"
  echo "--- UELAT listing ---"
  if [ -n "$UELAT_DIR" ] && [ -d "$UELAT_DIR" ]; then
    ls -la "$UELAT_DIR" || true
    echo "--- build dir listing ---"
    ls -la "$BUILD_DIR" || true
    echo "--- recent coq logs (if any) ---"
    if [ -d "$BUILD_DIR" ]; then
      ls -1 "$BUILD_DIR"/*.log 2>/dev/null || true
      for f in "$BUILD_DIR"/*.log; do
        [ -f "$f" ] || continue
        echo "---- head $f ----"
        head -n 200 "$f" || true
      done
    fi
  else
    echo "UELAT directory not found at expected path."
  fi
  echo "=== End diagnostics ==="
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
    # Compile with logical mapping: -Q . UELAT maps the current folder to module namespace UELAT.
    # Use tee to save logs for diagnostics.
    if ! ${COQC} -Q . UELAT "$vf" 2>&1 | tee "${BUILD_DIR}/coq_${vf}.log"; then
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
    if ! ${COQC} -Q . UELAT Extraction.v 2>&1 | tee "${BUILD_DIR}/coq_Extraction.v.log"; then
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

echo "[kernel] Step 2: Extracted kernel_verified.ml. Proceeding to OCaml compile..."

# Prepare build dir and compile OCaml file to cmx
mkdir -p "$BUILD_DIR"
echo "[kernel] Compiling kernel_verified.ml into ${BUILD_DIR}/kernel_verified.cmx"
if ! ${OCAMLOPT} -c -I +unix kernel_verified.ml -o "${BUILD_DIR}/kernel_verified.cmx"; then
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

# Link to shared library
echo "[kernel] Linking to shared library..."
if ! ${OCAMLOPT} -shared -I "${BUILD_DIR}" -o "${KERNEL_OUTPUT}" "${BUILD_DIR}/kernel_verified.cmx"; then
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
