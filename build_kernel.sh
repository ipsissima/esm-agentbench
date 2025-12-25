#!/bin/bash
##############################################################################
# build_kernel.sh: Build the Verified Certificate Kernel
#
# This script compiles the Coq formal proofs into an OCaml kernel,
# then builds a shared library (.so/.dylib) for Python to load via ctypes.
#
# Usage:
#   ./build_kernel.sh [clean]
#
# Environment:
#   COQC       - Coq compiler (default: coqc)
#   OCAMLOPT   - OCaml optimizing compiler (default: ocamlopt)
##############################################################################

set -e

# Configuration
UELAT_DIR="$(dirname "$0")/UELAT"
BUILD_DIR="${UELAT_DIR}/_build"
KERNEL_OUTPUT="${UELAT_DIR}/kernel_verified.so"

# Tools
COQC="${COQC:-coqc}"
OCAMLOPT="${OCAMLOPT:-ocamlopt}"
OCAMLFIND="${OCAMLFIND:-ocamlfind}"

# Check if running in clean mode
if [ "$1" = "clean" ]; then
    echo "[kernel] Cleaning build artifacts..."
    rm -rf "${BUILD_DIR}" "${KERNEL_OUTPUT}"
    echo "[kernel] Clean complete."
    exit 0
fi

# Step 1: Verify Coq files compile
echo "[kernel] Step 1: Verifying Coq proofs..."
cd "${UELAT_DIR}"

echo "  - Compiling spectral_bounds.v ..."
"${COQC}" -o /tmp/spectral_bounds.vo spectral_bounds.v

echo "  - Compiling CertificateCore.v ..."
"${COQC}" -I /tmp spectral_bounds.vo -o /tmp/CertificateCore.vo CertificateCore.v

echo "  - Compiling CertificateProofs.v ..."
"${COQC}" -I /tmp spectral_bounds.vo CertificateCore.vo -o /tmp/CertificateProofs.vo CertificateProofs.v

echo "  - Compiling Extraction.v ..."
"${COQC}" -I /tmp spectral_bounds.vo CertificateCore.vo -o /tmp/Extraction.vo Extraction.v

# Step 2: Extract to OCaml
echo "[kernel] Step 2: Extracting Coq to OCaml..."

# The extraction creates kernel_verified.ml
# This is handled by Coq's extraction mechanism in Extraction.v
# The extracted file should now exist in the UELAT directory

if [ ! -f "kernel_verified.ml" ]; then
    echo "[kernel] ERROR: Extraction failed - kernel_verified.ml not found"
    exit 1
fi

echo "  - Extracted kernel_verified.ml"

# Step 3: Compile OCaml to shared library
echo "[kernel] Step 3: Compiling OCaml kernel to shared library..."

mkdir -p "${BUILD_DIR}"

# Compile OCaml to object file
echo "  - Compiling OCaml object..."
"${OCAMLOPT}" -c -I +unix kernel_verified.ml -o "${BUILD_DIR}/kernel_verified.cmx"

# Create a C wrapper that exposes the kernel functions
echo "  - Creating C wrapper..."
cat > "${BUILD_DIR}/kernel_stub.c" << 'WRAPPER_END'
/* C wrapper for OCaml kernel functions */
#include <caml/mlvalues.h>
#include <caml/callback.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <stdio.h>

/* Initialize OCaml runtime */
void kernel_init(void) {
    static int initialized = 0;
    if (!initialized) {
        caml_startup(0);
        initialized = 1;
    }
}

/* Wrapper: compute_frobenius_norm_matrix *)
/* Wrapper: compute_residual_matrix *)
/* Wrapper: compute_bound *)
/* Wrapper: compute_certificate *)

/* These stubs provide C-callable interfaces to OCaml functions.
   The actual implementation uses ctypes on the Python side,
   calling OCaml functions directly via FFI. */

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
    kernel_init();

    /* Convert C arrays to OCaml lists *)
    /* Call OCaml kernel_compute_certificate *)
    /* Extract results back to C *)

    /* TODO: Implement full C<->OCaml marshalling *)
    /* For now, this is a placeholder */
    *out_residual = 0.0;
    *out_bound = 0.0;
}
WRAPPER_END

echo "  - Creating kernel_stub.c"

# Link to shared library
echo "  - Linking to shared library..."
"${OCAMLOPT}" -shared \
    -I "${BUILD_DIR}" \
    -o "${KERNEL_OUTPUT}" \
    kernel_verified.cmx

if [ -f "${KERNEL_OUTPUT}" ]; then
    echo "[kernel] SUCCESS: Kernel built at ${KERNEL_OUTPUT}"
    ls -lh "${KERNEL_OUTPUT}"
    exit 0
else
    echo "[kernel] ERROR: Failed to create shared library"
    exit 1
fi
