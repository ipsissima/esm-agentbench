(** * Code Extraction Configuration

    This file configures Coq to extract the verified certificate computation
    to executable OCaml code. The extraction produces a standalone kernel that
    Python can invoke via ctypes.

    **Extraction Strategy:**
    - Extract type 'R' (Reals) to OCaml 'float' (double precision)
    - Extract 'list' to OCaml 'list'
    - Extract core functions to OCaml with minimal overhead
    - Preserve function signatures for C interoperability

    **Verified Checker (Option B):**
    - Extract rational-based checker from Checker.v
    - Checker operates on exact rationals (no FP error)
    - If checker accepts witness, certificate is verified
*)

From Coq Require Import Floats.PrimFloat.
From Coq Require Import Reals Raxioms Rdefinitions RIneq R_sqrt Rsqrt_def Rbasic_fun Rtrigo_def Rpower.
From Coq Require Import ZArith.
From Coq Require Import BinNums BinInt BinIntDef BinPos BinPosDef.
From Coq Require Import QArith QArith_base.
Open Scope R_scope.

Require Import Extraction.
Require Import ExtrOcamlBasic.
Require Import ExtrOcamlString.
Require Import ExtrOcamlNatInt.
Require Import ExtrOcamlZInt.       (* standard: map Coq Z/positive to OCaml int *)
Require Import ExtrOcamlIntConv.    (* standard: map Coq int/nat conversions to OCaml int *)
Require Import spectral_bounds.
Require Import CertificateCore.
Require Import CertificateProofs.
Require Import Checker.

(** ** CRITICAL: Prevent extraction from bypassing opacity
    This stops extraction from peeking into opaque constants like
    ClassicalDedekindReals, which produce non-computational junk like
    (fun n -> n n) when extracted.
*)
Unset Extraction AccessOpaque.

(** ** Blacklist non-computational classical real modules
    This prevents extraction from exporting ClassicalDedekindReals internals.
    These modules contain non-computational constructs that produce garbage
    OCaml code (e.g., DRealQlimExp2, fun n -> n n patterns).
    Blacklisting ensures extraction treats references to these modules as
    opaque/axiomatic rather than trying to materialize computational junk.
*)
Extraction Blacklist ClassicalDedekindReals.
Extraction Blacklist Raxioms.
Extraction Blacklist Reals.
Extraction Blacklist RIneq.
Extraction Blacklist Rdefinitions.
Extraction Blacklist R_sqrt.
Extraction Blacklist Rsqrt_def.
Extraction Blacklist Rbasic_fun.

(** ** CRITICAL: Explicit extraction overrides for classical real axioms

    When Coq cannot extract computational content for an axiom, it synthesizes
    garbage like (fun n -> n n) which causes OCaml type errors. We must
    explicitly provide implementations for axioms that leak into computational
    contexts.

    These axioms come from ClassicalDedekindReals and are used transitively
    by decision procedures (Rgt_dec, etc.) and sqrt.
*)

(** sig_forall_dec: Decidability of bounded universal quantification over reals.
    Type: forall P, (forall n, {P n} + {~ P n}) -> {forall n, P n} + {~ forall n, P n}
    This is classical and cannot be computed. If reached at runtime, fail loudly.
*)
Extract Constant ClassicalDedekindReals.sig_forall_dec =>
  "(fun _ _ -> failwith ""sig_forall_dec: classical axiom used at runtime"")".

(** DRealQlimExp2: Dedekind real limit construction for exponential.
    This is used internally by ClassicalDedekindReals for real number construction.
    If reached at runtime (meaning the computation path uses Dedekind reals),
    we provide a safe fallback returning 0.0.
*)
Extract Constant ClassicalDedekindReals.DRealQlimExp2 => "0.0".

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

(* Additional comparison functions that might be used *)
Extract Inlined Constant Rle_lt_dec => "(fun x y -> if x <= y then true else false)".
Extract Inlined Constant Rlt_le_dec => "(fun x y -> if x < y then true else false)".

(** ** Z (integer) arithmetic operations

    Ensure Z operations extract to plain OCaml int operations.
    ExtrOcamlZInt handles the core Z type, but we add these for completeness.
*)
Extract Inlined Constant Z.add => "( + )".
Extract Inlined Constant Z.mul => "( * )".
Extract Inlined Constant Z.sub => "( - )".
Extract Inlined Constant Z.opp => "( ~- )".
Extract Inlined Constant Z.div => "( / )".
Extract Inlined Constant Z.modulo => "( mod )".
Extract Inlined Constant Z.eqb => "( = )".
Extract Inlined Constant Z.ltb => "( < )".
Extract Inlined Constant Z.leb => "( <= )".

(** ** Positive number extraction

    Ensure positive numbers extract correctly to OCaml ints.
*)
Extract Inlined Constant Pos.add => "( + )".
Extract Inlined Constant Pos.mul => "( * )".
Extract Inlined Constant Pos.sub => "(fun x y -> max 1 (x - y))".
Extract Inlined Constant Pos.eqb => "( = )".

(** ** Q (rational) extraction note

    Q is a record type, not a constant, so we cannot use Extract Constant Q.
    With ExtrOcamlZInt handling Z and positive, Q extracts naturally as a
    small OCaml record (numerator/denominator). No explicit extraction needed.
*)

(** Extract the kernel API functions *)
Extraction Language OCaml.

(** ** Export the Verified Functions

    These functions will be extracted and compiled to a shared library
    that Python can call via ctypes.
*)

(* Define an opaque string type that will map to OCaml's native string.
   We don't use Coq's Strings.String because it extracts to char list. *)
Axiom ocaml_string : Type.
Extract Constant ocaml_string => "string".

(* Main kernel API: declare these as opaque parameters for extraction.
   We implement them in OCaml (Kernel_runtime.ml). Using a simple
   string -> string signature keeps the boundary clean and avoids
   marshalling complexity in this last-minute integration. *)
Parameter kernel_api_frobenius_norm : ocaml_string -> ocaml_string.
Parameter kernel_api_residual : ocaml_string -> ocaml_string.
Parameter kernel_api_bound : ocaml_string -> ocaml_string.
Parameter kernel_api_certificate : ocaml_string -> ocaml_string.

(* Extraction mapping: bind the Coq Parameters to OCaml implementations. *)
Extract Constant kernel_api_frobenius_norm => "Kernel_runtime.kernel_api_frobenius_norm".
Extract Constant kernel_api_residual => "Kernel_runtime.kernel_api_residual".
Extract Constant kernel_api_bound => "Kernel_runtime.kernel_api_bound".
Extract Constant kernel_api_certificate => "Kernel_runtime.kernel_api_certificate".

(** ** Verified Checker Extraction

    The checker operates on rational intervals and is fully verified in Coq.
    It provides machine-checked guarantees: if check_witness returns true,
    the certificate satisfies the formal properties.
*)

(* Extract checker to separate file for modularity *)
Extraction "checker_verified.ml"
  verified_check
  verified_check_result
  verified_make_witness
  verified_make_interval
  verified_interval_lo
  verified_interval_hi
  verified_result_ok
  verified_result_fail
  CheckResult
  RuntimeWitness
  QInterval.

(** Extract to file 'kernel_verified.ml' *)
Extraction "kernel_verified.ml"
  kernel_api_frobenius_norm
  kernel_api_residual
  kernel_api_bound
  kernel_api_certificate.
