
(** val negb : bool -> bool **)

let negb = function
| true -> false
| false -> true

type comparison =
| Eq
| Lt
| Gt

module Pos =
 struct
  (** val succ : int -> int **)

  let rec succ = Stdlib.Int.succ

  (** val add_carry : int -> int -> int **)

  let rec add_carry x y =
    (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
      (fun p ->
      (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
        (fun q0 -> (fun p->1+2*p) (add_carry p q0))
        (fun q0 -> (fun p->2*p) (add_carry p q0))
        (fun _ -> (fun p->1+2*p) (succ p))
        y)
      (fun p ->
      (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
        (fun q0 -> (fun p->2*p) (add_carry p q0))
        (fun q0 -> (fun p->1+2*p) (( + ) p q0))
        (fun _ -> (fun p->2*p) (succ p))
        y)
      (fun _ ->
      (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
        (fun q0 -> (fun p->1+2*p) (succ q0))
        (fun q0 -> (fun p->2*p) (succ q0))
        (fun _ -> (fun p->1+2*p) 1)
        y)
      x

  (** val pred_double : int -> int **)

  let rec pred_double x =
    (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
      (fun p -> (fun p->1+2*p) ((fun p->2*p) p))
      (fun p -> (fun p->1+2*p) (pred_double p))
      (fun _ -> 1)
      x

  (** val compare_cont : comparison -> int -> int -> comparison **)

  let rec compare_cont = fun c x y -> if x=y then c else if x<y then Lt else Gt

  (** val compare : int -> int -> comparison **)

  let compare = fun x y -> if x=y then Eq else if x<y then Lt else Gt
 end

module Z =
 struct
  (** val double : int -> int **)

  let double x =
    (fun f0 fp fn z -> if z=0 then f0 () else if z>0 then fp z else fn (-z))
      (fun _ -> 0)
      (fun p -> ((fun p->2*p) p))
      (fun p -> (~-) ((fun p->2*p) p))
      x

  (** val succ_double : int -> int **)

  let succ_double x =
    (fun f0 fp fn z -> if z=0 then f0 () else if z>0 then fp z else fn (-z))
      (fun _ -> 1)
      (fun p -> ((fun p->1+2*p) p))
      (fun p -> (~-) (Pos.pred_double p))
      x

  (** val pred_double : int -> int **)

  let pred_double x =
    (fun f0 fp fn z -> if z=0 then f0 () else if z>0 then fp z else fn (-z))
      (fun _ -> (~-) 1)
      (fun p -> (Pos.pred_double p))
      (fun p -> (~-) ((fun p->1+2*p) p))
      x

  (** val pos_sub : int -> int -> int **)

  let rec pos_sub x y =
    (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
      (fun p ->
      (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
        (fun q0 -> double (pos_sub p q0))
        (fun q0 -> succ_double (pos_sub p q0))
        (fun _ -> ((fun p->2*p) p))
        y)
      (fun p ->
      (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
        (fun q0 -> pred_double (pos_sub p q0))
        (fun q0 -> double (pos_sub p q0))
        (fun _ -> (Pos.pred_double p))
        y)
      (fun _ ->
      (fun f2p1 f2p f1 p ->
  if p<=1 then f1 () else if p mod 2 = 0 then f2p (p/2) else f2p1 (p/2))
        (fun q0 -> (~-) ((fun p->2*p) q0))
        (fun q0 -> (~-) (Pos.pred_double q0))
        (fun _ -> 0)
        y)
      x

  (** val compare : int -> int -> comparison **)

  let compare = fun x y -> if x=y then Eq else if x<y then Lt else Gt
 end

type q = { qnum : int; qden : int }

(** val qle_bool : q -> q -> bool **)

let qle_bool x y =
  ( <= ) (( * ) x.qnum y.qden) (( * ) y.qnum x.qden)

(** val qplus : q -> q -> q **)

let qplus x y =
  { qnum = (( + ) (( * ) x.qnum y.qden) (( * ) y.qnum x.qden)); qden =
    (( * ) x.qden y.qden) }

(** val qmult : q -> q -> q **)

let qmult x y =
  { qnum = (( * ) x.qnum y.qnum); qden = (( * ) x.qden y.qden) }

(** val qinv : q -> q **)

let qinv x =
  (fun f0 fp fn z -> if z=0 then f0 () else if z>0 then fp z else fn (-z))
    (fun _ -> { qnum = 0; qden = 1 })
    (fun p -> { qnum = x.qden; qden = p })
    (fun p -> { qnum = ((~-) x.qden); qden = p })
    x.qnum

(** val qdiv : q -> q -> q **)

let qdiv x y =
  qmult x (qinv y)

type qInterval = { ival_lo : q; ival_hi : q }

type runtimeWitness = { rw_residual : qInterval; rw_bound : qInterval;
                        rw_tail_energy : q; rw_semantic_div : q;
                        rw_lipschitz : q; rw_frob_x1 : q; rw_frob_error : 
                        q }

(** val interval_valid : qInterval -> bool **)

let interval_valid i =
  qle_bool i.ival_lo i.ival_hi

(** val interval_nonneg : qInterval -> bool **)

let interval_nonneg i =
  qle_bool { qnum = 0; qden = 1 } i.ival_lo

(** val q_nonneg : q -> bool **)

let q_nonneg q0 =
  qle_bool { qnum = 0; qden = 1 } q0

(** val qlt_bool : q -> q -> bool **)

let qlt_bool q1 q2 =
  negb (qle_bool q2 q1)

(** val c_res_Q : q **)

let c_res_Q =
  { qnum = 1; qden = 1 }

(** val c_tail_Q : q **)

let c_tail_Q =
  { qnum = 1; qden = 1 }

(** val c_sem_Q : q **)

let c_sem_Q =
  { qnum = 1; qden = 1 }

(** val c_robust_Q : q **)

let c_robust_Q =
  { qnum = 1; qden = 1 }

(** val compute_formula_bound : q -> q -> q -> q -> q **)

let compute_formula_bound residual_hi tail sem lip =
  qplus
    (qplus (qplus (qmult c_res_Q residual_hi) (qmult c_tail_Q tail))
      (qmult c_sem_Q sem)) (qmult c_robust_Q lip)

(** val check_witness : runtimeWitness -> bool **)

let check_witness w =
  (&&)
    ((&&)
      ((&&)
        ((&&)
          ((&&)
            ((&&)
              ((&&)
                ((&&)
                  ((&&) (interval_valid w.rw_residual)
                    (interval_nonneg w.rw_residual))
                  (interval_valid w.rw_bound)) (interval_nonneg w.rw_bound))
              (q_nonneg w.rw_tail_energy)) (q_nonneg w.rw_semantic_div))
          (q_nonneg w.rw_lipschitz)) (q_nonneg w.rw_frob_x1))
      (q_nonneg w.rw_frob_error))
    (let formula_lo =
       compute_formula_bound w.rw_residual.ival_lo w.rw_tail_energy
         w.rw_semantic_div w.rw_lipschitz
     in
     (&&)
       ((&&) (qle_bool formula_lo w.rw_bound.ival_hi)
         (qle_bool w.rw_residual.ival_hi { qnum = ((fun p->2*p) 1); qden =
           1 }))
       (if qlt_bool { qnum = 1; qden = ((fun p->2*p) ((fun p->2*p)
             ((fun p->2*p) ((fun p->2*p) ((fun p->2*p) ((fun p->2*p)
             ((fun p->2*p) ((fun p->2*p) ((fun p->2*p) ((fun p->2*p)
             ((fun p->2*p) ((fun p->2*p) ((fun p->1+2*p) ((fun p->2*p)
             ((fun p->2*p) ((fun p->2*p) ((fun p->1+2*p) ((fun p->2*p)
             ((fun p->1+2*p) ((fun p->2*p) ((fun p->2*p) ((fun p->1+2*p)
             ((fun p->2*p) ((fun p->1+2*p) ((fun p->2*p) ((fun p->2*p)
             ((fun p->1+2*p) ((fun p->2*p) ((fun p->1+2*p) ((fun p->2*p)
             ((fun p->1+2*p) ((fun p->1+2*p) ((fun p->2*p) ((fun p->2*p)
             ((fun p->2*p) ((fun p->1+2*p) ((fun p->2*p) ((fun p->1+2*p)
             ((fun p->1+2*p) 1))))))))))))))))))))))))))))))))))))))) }
             w.rw_frob_x1
        then qle_bool (qdiv w.rw_frob_error w.rw_frob_x1)
               w.rw_residual.ival_hi
        else true))

(** val make_interval : q -> q -> qInterval **)

let make_interval lo hi =
  { ival_lo = lo; ival_hi = hi }

(** val build_witness :
    q -> q -> q -> q -> q -> q -> q -> q -> q -> runtimeWitness **)

let build_witness res_lo res_hi bound_lo bound_hi tail_energy semantic_div lipschitz frob_x1 frob_error =
  { rw_residual = (make_interval res_lo res_hi); rw_bound =
    (make_interval bound_lo bound_hi); rw_tail_energy = tail_energy;
    rw_semantic_div = semantic_div; rw_lipschitz = lipschitz; rw_frob_x1 =
    frob_x1; rw_frob_error = frob_error }

type checkResult =
| CheckOK
| CheckFail

(** val check_witness_result : runtimeWitness -> checkResult **)

let check_witness_result w =
  if check_witness w then CheckOK else CheckFail

(** val verified_check : runtimeWitness -> bool **)

let verified_check =
  check_witness

(** val verified_check_result : runtimeWitness -> checkResult **)

let verified_check_result =
  check_witness_result

(** val verified_make_witness :
    q -> q -> q -> q -> q -> q -> q -> q -> q -> runtimeWitness **)

let verified_make_witness =
  build_witness

(** val verified_make_interval : q -> q -> qInterval **)

let verified_make_interval =
  make_interval

(** val verified_interval_lo : qInterval -> q **)

let verified_interval_lo q0 =
  q0.ival_lo

(** val verified_interval_hi : qInterval -> q **)

let verified_interval_hi q0 =
  q0.ival_hi

(** val verified_result_ok : checkResult **)

let verified_result_ok =
  CheckOK

(** val verified_result_fail : checkResult **)

let verified_result_fail =
  CheckFail
