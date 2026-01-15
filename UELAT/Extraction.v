(** * Code Extraction Configuration

    This file configures Coq to extract the verified certificate computation
    to executable OCaml code. The extraction produces a standalone kernel that
    Python can invoke via ctypes.

    **Extraction Strategy:**
    - Extract type 'R' (Reals) to OCaml 'float' (double precision)
    - Extract 'list' to OCaml 'list'
    - Extract core functions to OCaml with minimal overhead
    - Preserve function signatures for C interoperability
*)

From Coq Require Import Floats.PrimFloat.
Require Import Extraction.
Require Import ExtrOcamlBasic.
Require Import ExtrOcamlString.
Require Import ExtrOcamlNatInt.
Require Import spectral_bounds.
Require Import CertificateCore.
Require Import CertificateProofs.

(** ** OCaml Code Generation Settings *)

(** Use native OCaml floats (double precision) for R *)
(* Map the PrimFloat primitive float type to OCaml's native float. *)
Extract Constant PrimFloat.float => "float".

(** Extract the kernel API functions *)
Extraction Language OCaml.

(** ** Export the Verified Functions

    These functions will be extracted and compiled to a shared library
    that Python can call via ctypes.
*)

(* Main kernel function: accepts witness matrices and parameters *)
Definition kernel_api_frobenius_norm := frobenius_norm_squared.
Definition kernel_api_residual := compute_residual.
Definition kernel_api_bound := compute_theoretical_bound.
Definition kernel_api_certificate := kernel_compute_certificate.

(** Extract to file 'kernel_verified.ml' *)
Extraction "kernel_verified.ml"
  kernel_api_frobenius_norm
  kernel_api_residual
  kernel_api_bound
  kernel_api_certificate.
