(** * Core Certificate Computation (Verified Kernel)

    This module provides the formally verified implementation of:
    1. Frobenius norm computation
    2. Residual calculation
    3. Theoretical bound computation

    All operations are defined over real numbers (R) and extracted to OCaml
    for high-precision floating-point computation.

    **Witness-Checker Architecture:**
    - Python (unverified): Computes SVD of X, derives matrices X0, X1, A
    - Kernel (verified): Computes residual and bound from these matrices
*)

Require Import Reals.
Require Import Raxioms.
Require Import Lra.
Require Import List.
Import ListNotations.
Require Import spectral_bounds.

Open Scope R_scope.

(** ** Data Type: Matrix as nested lists

    For simplicity and extractability, we represent matrices as lists of lists.
    A matrix of shape (m, n) is List R with m rows, each of length n.
*)

Definition Matrix := list (list R).

(** ** Vector Operations *)

(** Dot product of two vectors *)
Definition dot_product (v1 v2 : list R) : R :=
  fold_right (fun '(x, y) acc => x * y + acc) 0 (combine v1 v2).

(** Vector subtraction *)
Definition vec_sub (v1 v2 : list R) : list R :=
  map (fun '(x, y) => x - y) (combine v1 v2).

(** Vector norm squared *)
Definition vec_norm_sq (v : list R) : R :=
  fold_right (fun x acc => x * x + acc) 0 v.

(** ** Frobenius Norm Helper Functions *)

(** Sum of squares of all elements in a vector *)
Definition sum_squares (v : list R) : R :=
  fold_right (fun x acc => x * x + acc) 0 v.

(** Frobenius norm squared: sum of all element squares *)
Definition frobenius_norm_squared (M : Matrix) : R :=
  fold_right (fun row acc => sum_squares row + acc) 0 M.

(** ** Matrix Algebra Operations *)

(** Get the i-th row of a matrix (0-indexed) *)
Definition get_row (M : Matrix) (i : nat) : list R :=
  match nth_error M i with
  | Some row => row
  | None => []
  end.

(** Get the j-th column of a matrix (0-indexed) *)
Definition get_column (M : Matrix) (j : nat) : list R :=
  map (fun row =>
    match nth_error row j with
    | Some x => x
    | None => 0
    end
  ) M.

(** Transpose a matrix *)
Definition transpose (M : Matrix) : Matrix :=
  match M with
  | [] => []
  | firstrow :: _ =>
      let ncols := length firstrow in
      List.map (fun j => get_column M j) (List.seq 0 ncols)
  end.

(** Matrix multiplication: A (m x n) * B (n x p) = C (m x p)
    Result[i,j] = dot_product(Row_i(A), Column_j(B))
*)
Definition matrix_mult (A B : Matrix) : Matrix :=
  let B_T := transpose B in
  map (fun row_a =>
    map (fun col_b => dot_product row_a col_b) B_T
  ) A.

(** Matrix subtraction: A - B element-wise *)
Definition matrix_sub (A B : Matrix) : Matrix :=
  map (fun '(row_a, row_b) => vec_sub row_a row_b) (combine A B).

(** ** Residual Computation

    Given matrices X0, X1, and operator A, compute:
      residual = ||X1 - A * X0||_F / ||X1||_F

    This measures the prediction error in the reduced space.
    Defined as a ratio to be scale-invariant.
*)

Definition compute_residual (X0 X1 : Matrix) (A : Matrix) : R :=
  let AX0 := matrix_mult A X0 in
  let error := matrix_sub X1 AX0 in
  let error_norm_sq := frobenius_norm_squared error in
  let x1_norm_sq := frobenius_norm_squared X1 in
  let eps := 1e-12 in
  if Rgt_dec x1_norm_sq eps then
    sqrt (error_norm_sq / x1_norm_sq)
  else
    0.

(** ** Theoretical Bound Computation

    Combine all terms with verified constants:
      theoretical_bound = C_res * residual
                        + C_tail * tail_energy
                        + C_sem * semantic_divergence
                        + C_robust * lipschitz_margin
*)

Definition compute_bound
  (residual tail_energy semantic_divergence lipschitz_margin : R)
  (c_res c_tail c_sem c_robust : R) : R :=
  c_res * residual +
  c_tail * tail_energy +
  c_sem * semantic_divergence +
  c_robust * lipschitz_margin.

(** ** Verification: Bound Satisfies Axioms *)

(** Lemma: The computed bound is non-negative for non-negative inputs *)
Lemma computed_bound_safe :
  forall r t s l cr ct cs cb : R,
  0 <= r -> 0 <= t -> 0 <= s -> 0 <= l ->
  0 < cr -> 0 < ct -> 0 < cs -> 0 < cb ->
  0 <= compute_bound r t s l cr ct cs cb.
Proof.
  intros r t s l cr ct cs cb Hr Ht Hs Hl Hcr Hct Hcs Hcb.
  unfold compute_bound.
  apply Rplus_le_le_0_compat.
  apply Rplus_le_le_0_compat.
  apply Rplus_le_le_0_compat.
  - apply Rmult_le_pos.
    + left. exact Hcr.
    + exact Hr.
  - apply Rmult_le_pos.
    + left. exact Hct.
    + exact Ht.
  - apply Rmult_le_pos.
    + left. exact Hcs.
    + exact Hs.
  - apply Rmult_le_pos.
    + left. exact Hcb.
    + exact Hl.
Qed.

(** Lemma: The computed bound increases monotonically with residual *)
Lemma computed_bound_monotone_residual :
  forall r1 r2 t s l cr ct cs cb : R,
  r1 <= r2 -> 0 < cr ->
  compute_bound r1 t s l cr ct cs cb <=
  compute_bound r2 t s l cr ct cs cb.
Proof.
  intros r1 r2 t s l cr ct cs cb Hr Hcr.
  unfold compute_bound.
  apply Rplus_le_compat_r.
  apply Rplus_le_compat_r.
  apply Rplus_le_compat_r.
  apply Rmult_le_compat_l.
  - left. exact Hcr.
  - exact Hr.
Qed.

(** Lemma: The computed bound respects the axiomatized constant bounds *)
Lemma computed_bound_uses_bounded_constants :
  forall r t s l : R,
  0 <= r -> 0 <= t -> 0 <= s -> 0 <= l ->
  compute_bound r t s l C_res_value C_tail_value C_sem_value C_robust_value <=
  2.0 * (r + t + s + l).
Proof.
  intros r t s l Hr Ht Hs Hl.
  unfold compute_bound, C_res_value, C_tail_value, C_sem_value, C_robust_value.
  ring_simplify.
  lra.
Qed.

(** ** Extraction-Ready Interface

    The following are the public functions that will be extracted to OCaml.
    They represent the "Verified Kernel" API.
*)

(** Wrapper: compute residual and bound from witness matrices *)
Definition kernel_compute_certificate
  (X0 X1 : Matrix) (A : Matrix)
  (tail_energy semantic_divergence lipschitz_margin : R) : R * R :=
  let residual := compute_residual X0 X1 A in
  let bound := compute_bound residual tail_energy
    semantic_divergence lipschitz_margin
    C_res_value C_tail_value C_sem_value C_robust_value in
  (residual, bound).

(** All exports for OCaml extraction *)
Definition frobenius_norm_of_matrix := frobenius_norm_squared.
Definition compute_residual_from_matrices := compute_residual.
Definition compute_theoretical_bound := compute_bound.
Definition kernel_certificate_api := kernel_compute_certificate.
