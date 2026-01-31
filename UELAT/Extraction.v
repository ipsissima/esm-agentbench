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
From Coq Require Import Reals Raxioms Rdefinitions RIneq R_sqrt Rsqrt_def Rbasic_fun Rtrigo_def Rpower.
From Coq Require Import ZArith.
Open Scope R_scope.

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

(* The R type itself - use Extract Constant for type axioms (not Inlined) *)
Extract Constant R => "float".

(* Real number constants *)
Extract Inlined Constant R0 => "0.0".
Extract Inlined Constant R1 => "1.0".

(* Basic arithmetic operations *)
Extract Inlined Constant Rplus => "( +. )".
Extract Inlined Constant Rmult => "( *. )".
Extract Inlined Constant Ropp => "( ~-. )".
Extract Inlined Constant Rinv => "(fun x -> 1.0 /. x)".

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
Extract Constant total_order_T => "fun x y ->
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

(* Mathematical functions *)
Extract Inlined Constant sqrt => "Float.sqrt".

(* Other transcendental functions *)
Extract Inlined Constant Rabs => "Float.abs".
Extract Inlined Constant exp => "Float.exp".
Extract Inlined Constant ln => "Float.log".
Extract Inlined Constant sin => "Float.sin".
Extract Inlined Constant cos => "Float.cos".

(* Power function *)
Extract Inlined Constant pow => "(fun x n -> x ** (Float.of_int n))".
Extract Inlined Constant Rpower => "Float.pow".

(* Integer to real conversion *)
Extract Inlined Constant IZR => "Float.of_int".
Extract Inlined Constant INR => "Float.of_int".

(* Coq's up function (ceiling) used in some proofs *)
Extract Inlined Constant up => "fun x -> int_of_float (Float.ceil x)".

(* archimed axiom - provides Archimedean property, used in some real computations *)
Extract Constant archimed => "fun r -> ((), ())".

(* completeness axiom - supremum existence, should be erased but add just in case *)
Extract Constant completeness => "fun _ _ -> 0.0".

(** ** Additional numeric conversions and power functions

    These handle the expansion of numeric literals like 1e-12 which may use
    various Coq constructs for rational/integer to real conversion.
*)

(* Z operations for integer literals in R scope *)
Extract Inlined Constant Z.of_nat => "fun n -> n".
Extract Inlined Constant Z.to_nat => "fun z -> max 0 z".
Extract Inlined Constant Z.abs_nat => "fun z -> abs z".

(* Power functions - both nat and Z based *)
Extract Inlined Constant powerRZ => "(fun x z -> x ** (Float.of_int z))".
Extract Inlined Constant IPR => "Float.of_int".
Extract Inlined Constant IPR_2 => "(fun p -> Float.of_int (2 * p))".
Extract Inlined Constant IZR_POS => "(fun p -> Float.of_int p)".

(* Additional comparison functions that might be used *)
Extract Inlined Constant Rle_lt_dec => "(fun x y -> if x <= y then true else false)".
Extract Inlined Constant Rlt_le_dec => "(fun x y -> if x < y then true else false)".

(* Rcompare for trichotomy - used internally *)
Extract Constant Rcompare => "fun x y ->
  if x < y then Lt
  else if x = y then Eq
  else Gt".

(* Bool to sumbool conversion if needed *)
Extract Inlined Constant Sumbool.sumbool_of_bool => "fun b -> b".

(* Positive number operations - used in numeric literal expansions *)
Extract Inlined Constant Pos.of_nat => "fun n -> max 1 n".
Extract Inlined Constant Pos.to_nat => "fun p -> p".
Extract Inlined Constant Pos.of_succ_nat => "fun n -> n + 1".

(* mult_IZR and plus_IZR for numeric operations *)
Extract Inlined Constant mult_IZR => "(fun z1 z2 -> Float.of_int z1 *. Float.of_int z2)".
Extract Inlined Constant plus_IZR => "(fun z1 z2 -> Float.of_int z1 +. Float.of_int z2)".
Extract Inlined Constant opp_IZR => "(fun z -> -. Float.of_int z)".

(* Rlt_bool and other boolean comparisons *)
Extract Inlined Constant Rlt_bool => "(fun x y -> x < y)".
Extract Inlined Constant Rle_bool => "(fun x y -> x <= y)".
Extract Inlined Constant Req_bool => "(fun x y -> x = y)".

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
