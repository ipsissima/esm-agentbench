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

(** ** Extract Coq Real Numbers (R) to OCaml float

    Coq's R type is axiomatically defined and cannot be executed directly.
    We extract it to OCaml's native float (IEEE 754 double precision).
    This sacrifices the axiomatic guarantees for computability.
*)

(* The R type itself *)
Extract Inlined Constant R => "float".

(* Real number constants *)
Extract Inlined Constant R0 => "0.0".
Extract Inlined Constant R1 => "1.0".

(* Basic arithmetic operations *)
Extract Inlined Constant Rplus => "( +. )".
Extract Inlined Constant Rminus => "( -. )".
Extract Inlined Constant Rmult => "( *. )".
Extract Inlined Constant Rdiv => "( /. )".
Extract Inlined Constant Ropp => "( ~-. )".
Extract Inlined Constant Rinv => "(fun x -> 1.0 /. x)".

(* Comparison operations - return bool for extracted code *)
Extract Inlined Constant Rlt_dec => "(fun x y -> x < y)".
Extract Inlined Constant Rgt_dec => "(fun x y -> x > y)".
Extract Inlined Constant Rle_dec => "(fun x y -> x <= y)".
Extract Inlined Constant Rge_dec => "(fun x y -> x >= y)".
Extract Inlined Constant Req_dec => "(fun x y -> x = y)".

(* Mathematical functions from Reals *)
Extract Inlined Constant sqrt => "Float.sqrt".
Extract Inlined Constant Rabs => "Float.abs".
Extract Inlined Constant exp => "Float.exp".
Extract Inlined Constant ln => "Float.log".
Extract Inlined Constant sin => "Float.sin".
Extract Inlined Constant cos => "Float.cos".

(* Power function *)
Extract Inlined Constant pow => "(fun x n -> x ** (Float.of_int n))".
Extract Inlined Constant Rpower => "Float.pow".

(* IZR: integer to real conversion *)
Extract Inlined Constant IZR => "Float.of_int".
Extract Inlined Constant INR => "Float.of_int".

(* Epsilon for numerical stability - use OCaml float literal *)
Extract Inlined Constant Rdefinitions.IZR => "Float.of_int".

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
