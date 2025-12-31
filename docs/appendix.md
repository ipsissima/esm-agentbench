# Spectral Certificate Math Appendix

This appendix documents the finite-rank spectral certificate used in the demo pipeline. It links the numerical certificate to a conservative theoretical bound and references the official competition deck at `file:///mnt/data/agentbeats-competition-info-session-deck.pdf` for context.

## Assumptions and setup
- Embeddings: a sequence of vectors $x_t \in \mathbb{R}^d$ for $t=0,\ldots,T-1$.
- Affine drift: handled by augmenting embeddings with a bias term $1$, yielding $\tilde{x}_t = [x_t;1]$ so linear proxies can represent affine dynamics without stationarity.
- Finite-rank proxy: SVD projects $\tilde{x}_t$ onto an $r$-dimensional subspace $U \in \mathbb{R}^{(d+1)\times r}$ with orthonormal columns. The retained variance fraction is $\text{pca\_explained} = \sum_{i=1}^r \sigma_i^2 / \sum_{j} \sigma_j^2$ where $\sigma_i$ are singular values.
- Linear dynamics: in the reduced space $z_t = U^\top \tilde{x}_t$, we fit a linear map $A$ minimizing $\lVert z_{t+1} - A z_t \rVert_F^2$.

## Quantities
- **Residual**: $\text{residual} = \frac{\lVert Z_1 - A Z_0 \rVert_F}{\lVert Z_1 \rVert_F + \epsilon}$ with $Z_0 = [z_0,\ldots,z_{T-2}]$, $Z_1 = [z_1,\ldots,z_{T-1}]$, and $\epsilon=10^{-12}$ for stability.
- **Tail energy**: $\text{tail\_energy} = 1 - \text{pca\_explained}$ (clipped to $[0,1]$), the fraction of variance lost in rank truncation.
- **Theoretical bound**: $\text{theoretical\_bound} = \text{residual} + \text{tail\_energy}$.
- **Spectral terms**: $\sigma_{\max}$ is the leading singular value; $\text{singular\_gap}$ is the difference between the two largest singular values (Wedin stability margin).

## Why residual + tail is conservative
Let $X_0, X_1$ be the augmented trajectories and $P = UU^\top$ the PCA projector. Decompose $X_1 = PX_1 + (I-P)X_1$. The fitted $A$ satisfies
\[
\lVert PX_1 - A P X_0 \rVert_F \le \lVert X_1 - A P X_0 \rVert_F = \text{residual}\cdot \lVert X_1 \rVert_F.
\]
Meanwhile, the discarded component obeys $\lVert(I-P)X_1\rVert_F^2 = (1-\text{pca\_explained})\lVert X_1 \rVert_F^2$. By the triangle inequality,
\[
\frac{\lVert X_1 - A P X_0 \rVert_F}{\lVert X_1 \rVert_F} \le \text{residual} + (1-\text{pca\_explained}),
\]
so $\text{theoretical\_bound}$ upper-bounds the normalized Frobenius reconstruction error of the finite-rank proxy. The affine augmentation ensures drift does not invalidate the inequality because $P$ spans the bias direction.

## Interpreting the certificate
- **sigma\_max** is the leading singular value; provides stability proxy via Wedin's theorem.
- **singular\_gap** quantifies separation between dominant modes; larger gaps imply stable singular subspaces.
- **residual** captures fit error within the retained subspace; lower is better.
- **tail\_energy** quantifies variance lost in rank truncation.
- **theoretical\_bound** combines both sources and is the conservative scalar a judge or verifier should compare against a tolerance for finite-rank approximation.

## Sandboxing note
Unit tests run via `pytest -q` inside a temporary directory with a short timeout, and without `shell=True`. For stronger isolation, judges can execute the same flow inside the provided Docker image using `docker run --network none esm-agentbench:ci` to disable external access.

### Coq / UELAT bridge and constants
The formal UELAT lemma bounding the finite-rank reconstruction error takes the form
\(\|X_1 - \hat{A}PX_0\|/\|X_1\| \leq C_{\text{tail}}\,\text{tail} + C_{\text{res}}\,\text{residual}\),
where ``tail`` is the SVD variance discarded and ``residual`` is the in-subspace linear fit error.
In the current certificate code, ``theoretical_bound`` already implements ``residual + tail``.
Therefore the runtime *guaranteed* bound is instantiated as
``guaranteed_bound = C_tail * tail_energy + C_res * residual`` with Coq-exported constants.
When the lemma states the coefficients are exactly 1, we set ``C_tail = 1`` and ``C_res = 1``;
other extracted constants will automatically override these defaults via the bridge.

To extract numeric constants from Coq manually:
```
coqtop -batch -quiet -l path/to/ulelat.v -eval 'Print C_tail.' -eval 'Print C_res.' \
  2>&1 | sed -n 's/.*C_tail *= *\([^ ]*\).*/C_tail \1/p; s/.*C_res *= *\([^ ]*\).*/C_res \1/p'
```
The helper ``tools/generate_uelat_constants_from_coq.sh`` automates this flow:
```
./tools/generate_uelat_constants_from_coq.sh path/to/ulelat.v certificates/uelat_constants.json C_tail C_res
```

Annotated Coq references (placeholders to be replaced with the project's concrete files):
- `ULELAT/lemmas/finite_rank_truncation.v` — Formalization of the finite-rank truncation lemma proving constants `C_tail` and `C_res` under assumptions on SVD variance and trajectory residuals.
- `ULELAT/analysis/constants_export.v` — Utility lemmas that `Print` the numeric instantiations of the constants for extraction into JSON.

Verification workflow:
1. Export constants with ``tools/generate_uelat_constants_from_coq.sh`` (or manually with ``coqtop``) to create ``uelat_constants.json``.
2. Load them at runtime via ``certificates/uelat_bridge.py`` to sanitize and expose ``C_tail`` / ``C_res``.
3. Run ``python certificates/verify_bound.py --constants certificates/uelat_constants.json --datasets all --T 40 --d 8 --r-values 1,2,4 --trials 3``.
   The script compares the runtime ``theoretical_bound`` against the formal ``guaranteed_bound`` and reports ``OK`` if the conservative inequality holds for all trials.
