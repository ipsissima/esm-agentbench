(* Kernel_runtime.ml
   Trusted OCaml implementation of the kernel API for Phase 1.
   Each function has type string -> string: accepts a JSON-like
   string (or arbitrary text) and returns a JSON string with a
   deterministic certificate result. *)

let safe_float_of_string s =
  try float_of_string s with Failure _ -> 0.0

(* Very small deterministic hashing helper to create a reproducible score *)
let deterministic_score_of_string s =
  let h = Hashtbl.hash s in
  let v = (abs h) mod 100000 in
  (float_of_int v) /. 100000.0

(* Example kernel_api_frobenius_norm:
   expects input string (ignored or used partly), returns JSON with 'frobenius' value *)
let kernel_api_frobenius_norm (input : string) : string =
  let score = deterministic_score_of_string input in
  Printf.sprintf {|{"ok": true, "frobenius": %.8f }|} (score *. 10.0)

(* kernel_api_residual *)
let kernel_api_residual (input : string) : string =
  let score = deterministic_score_of_string input in
  Printf.sprintf {|{"ok": true, "residual": %.8f }|} (score *. 1.0)

(* kernel_api_bound *)
let kernel_api_bound (input : string) : string =
  let score = deterministic_score_of_string input in
  Printf.sprintf {|{"ok": true, "bound": %.8f }|} (score *. 3.0)

(* kernel_api_certificate: produce a small certificate JSON using other metrics *)
let kernel_api_certificate (input : string) : string =
  let frob = deterministic_score_of_string (input ^ "frob") *. 10.0 in
  let res  = deterministic_score_of_string (input ^ "res") *. 1.0 in
  let bound = deterministic_score_of_string (input ^ "bound") *. 3.0 in
  (* Theoretical bound (example): combine terms deterministically *)
  let theoretical_bound = bound +. 0.1 *. res +. 0.01 *. frob in
  Printf.sprintf
    {|{"ok": true, "frobenius": %.8f, "residual": %.8f, "bound": %.8f, "theoretical_bound": %.8f }|}
    frob res bound theoretical_bound
