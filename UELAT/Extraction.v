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

    IMPORTANT: We must extract ALL axioms that the code depends on,
    including indirect dependencies through the standard library.
*)

(* The R type itself - use fully qualified path *)
Extract Inlined Constant Rdefinitions.R => "float".

(* Real number constants - fully qualified *)
Extract Inlined Constant Rdefinitions.R0 => "0.0".
Extract Inlined Constant Rdefinitions.R1 => "1.0".

(* Basic arithmetic operations from Rdefinitions *)
Extract Inlined Constant Rdefinitions.Rplus => "( +. )".
Extract Inlined Constant Rdefinitions.Rmult => "( *. )".
Extract Inlined Constant Rdefinitions.Ropp => "( ~-. )".
Extract Inlined Constant Rdefinitions.Rinv => "(fun x -> 1.0 /. x)".

(* Derived operations from Raxioms/RIneq *)
Extract Inlined Constant Rminus => "( -. )".
Extract Inlined Constant Rdiv => "( /. )".

(** ** Critical Axiom: total_order_T

    This is the fundamental axiom that Rgt_dec, Rlt_dec, etc. depend on.
    It provides decidable trichotomy for real numbers.

    Type: forall r1 r2 : R, {r1 < r2} + {r1 = r2} + {r1 > r2}
    This is: sumor (sumbool) (Prop)

    With ExtrOcamlBasic:
    - sumbool extracts to bool (left=true, right=false)
    - sumor extracts to option (inleft=Some, inright=None)

    So total_order_T returns option bool:
    - r1 < r2  -> Some true   (inleft (left _))
    - r1 = r2  -> Some false  (inleft (right _))
    - r1 > r2  -> None        (inright _)
*)
Extract Constant Raxioms.total_order_T => "fun x y ->
  if x < y then Some true
  else if x = y then Some false
  else None".

(* Comparison decision procedures - these use total_order_T internally
   but we can override them directly for efficiency *)
Extract Inlined Constant Rlt_dec => "(fun x y -> if x < y then true else false)".
Extract Inlined Constant Rgt_dec => "(fun x y -> if x > y then true else false)".
Extract Inlined Constant Rle_dec => "(fun x y -> if x <= y then true else false)".
Extract Inlined Constant Rge_dec => "(fun x y -> if x >= y then true else false)".
Extract Inlined Constant Req_dec => "(fun x y -> if x = y then true else false)".
(* Req_EM_T: {r1 = r2} + {r1 <> r2} - extracts to bool via sumbool *)
Extract Inlined Constant Req_EM_T => "(fun x y -> x = y)".

(* Mathematical functions from R_sqrt and other modules *)
Extract Inlined Constant R_sqrt.sqrt => "Float.sqrt".
Extract Inlined Constant sqrt => "Float.sqrt".
Extract Inlined Constant Rabs => "Float.abs".
Extract Inlined Constant exp => "Float.exp".
Extract Inlined Constant ln => "Float.log".
Extract Inlined Constant sin => "Float.sin".
Extract Inlined Constant cos => "Float.cos".

(* Power function *)
Extract Inlined Constant pow => "(fun x n -> x ** (Float.of_int n))".
Extract Inlined Constant Rpower => "Float.pow".

(* Integer to real conversion - multiple possible paths *)
Extract Inlined Constant Rdefinitions.IZR => "Float.of_int".
Extract Inlined Constant RIneq.IZR => "Float.of_int".
Extract Inlined Constant IZR => "Float.of_int".
Extract Inlined Constant INR => "Float.of_int".

(* Coq's up function (ceiling) used in some proofs *)
Extract Inlined Constant Raxioms.up => "fun x -> int_of_float (Float.ceil x)".

(* archimed axiom - provides Archimedean property, used in some real computations *)
Extract Constant Raxioms.archimed => "fun r -> ((), ())".

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
