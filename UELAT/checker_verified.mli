
val negb : bool -> bool

type comparison =
| Eq
| Lt
| Gt

module Pos :
 sig
  val succ : int -> int

  val add_carry : int -> int -> int

  val pred_double : int -> int

  val compare_cont : comparison -> int -> int -> comparison

  val compare : int -> int -> comparison
 end

module Z :
 sig
  val double : int -> int

  val succ_double : int -> int

  val pred_double : int -> int

  val pos_sub : int -> int -> int

  val compare : int -> int -> comparison
 end

type q = { qnum : int; qden : int }

val qle_bool : q -> q -> bool

val qplus : q -> q -> q

val qmult : q -> q -> q

val qinv : q -> q

val qdiv : q -> q -> q

type qInterval = { ival_lo : q; ival_hi : q }

type runtimeWitness = { rw_residual : qInterval; rw_bound : qInterval;
                        rw_tail_energy : q; rw_semantic_div : q;
                        rw_lipschitz : q; rw_frob_x1 : q; rw_frob_error : 
                        q }

val interval_valid : qInterval -> bool

val interval_nonneg : qInterval -> bool

val q_nonneg : q -> bool

val qlt_bool : q -> q -> bool

val c_res_Q : q

val c_tail_Q : q

val c_sem_Q : q

val c_robust_Q : q

val compute_formula_bound : q -> q -> q -> q -> q

val check_witness : runtimeWitness -> bool

val make_interval : q -> q -> qInterval

val build_witness :
  q -> q -> q -> q -> q -> q -> q -> q -> q -> runtimeWitness

type checkResult =
| CheckOK
| CheckFail

val check_witness_result : runtimeWitness -> checkResult

val verified_check : runtimeWitness -> bool

val verified_check_result : runtimeWitness -> checkResult

val verified_make_witness :
  q -> q -> q -> q -> q -> q -> q -> q -> q -> runtimeWitness

val verified_make_interval : q -> q -> qInterval

val verified_interval_lo : qInterval -> q

val verified_interval_hi : qInterval -> q

val verified_result_ok : checkResult

val verified_result_fail : checkResult
