(** OCaml main module for kernel callback registration

    This file registers the extracted kernel functions with the OCaml
    callback mechanism so they can be called from C via caml_named_value.

    Without this registration, caml_named_value("kernel_api_certificate")
    returns NULL, causing a SIGSEGV in the C wrapper.
*)

(* Import the extracted kernel functions *)
open Kernel_verified

(* Register callbacks at startup *)
let () =
  Callback.register "kernel_api_certificate" kernel_api_certificate;
  Callback.register "kernel_api_frobenius_norm" kernel_api_frobenius_norm;
  Callback.register "kernel_api_residual" kernel_api_residual;
  Callback.register "kernel_api_bound" kernel_api_bound
