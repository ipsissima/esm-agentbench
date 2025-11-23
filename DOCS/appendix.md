# Spectral Certificate Math Appendix

This appendix documents the finite-rank spectral certificate used in the demo pipeline. It links the numerical certificate to a conservative theoretical bound and references the official competition deck at `file:///mnt/data/agentbeats-competition-info-session-deck.pdf` for context.

## Assumptions and setup
- Embeddings: a sequence of vectors $x_t \in \mathbb{R}^d$ for $t=0,\ldots,T-1$.
- Affine drift: handled by augmenting embeddings with a bias term $1$, yielding $\tilde{x}_t = [x_t;1]$ so linear proxies can represent affine dynamics without stationarity.
- Finite-rank proxy: PCA projects $\tilde{x}_t$ onto an $r$-dimensional subspace $U \in \mathbb{R}^{(d+1)\times r}$ with orthonormal columns. The retained variance fraction is $\text{pca\_explained} = \sum_{i=1}^r \lambda_i / \sum_{j} \lambda_j$ where $\lambda_i$ are PCA eigenvalues.
- Koopman approximation: in the reduced space $z_t = U^\top \tilde{x}_t$, we fit a linear map $A$ minimizing $\lVert z_{t+1} - A z_t \rVert_F^2$.

## Quantities
- **Residual**: $\text{residual} = \frac{\lVert Z_1 - A Z_0 \rVert_F}{\lVert Z_1 \rVert_F + \epsilon}$ with $Z_0 = [z_0,\ldots,z_{T-2}]$, $Z_1 = [z_1,\ldots,z_{T-1}]$, and $\epsilon=10^{-12}$ for stability.
- **PCA tail estimate**: $\text{pca\_tail\_estimate} = 1 - \text{pca\_explained}$ (clipped to $[0,1]$), the fraction of variance discarded by PCA.
- **Theoretical bound**: $\text{theoretical\_bound} = \text{residual} + \text{pca\_tail\_estimate}$.
- **Spectral terms**: $\text{max\_eig}$ is the largest eigenvalue magnitude of $A$; $\text{spectral\_gap}$ is the difference between the two largest magnitudes.

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
- **max\_eig < 1** suggests contractive dynamics in the reduced space; values above 1 indicate potential instability.
- **spectral\_gap** highlights dominant modes; larger gaps imply a leading eigen-direction.
- **residual** captures fit error within the retained subspace; lower is better.
- **pca\_tail\_estimate** quantifies variance omitted by projection.
- **theoretical\_bound** combines both sources and is the conservative scalar a judge or verifier should compare against a tolerance for finite-rank approximation.

## Sandboxing note
Unit tests run via `pytest -q` inside a temporary directory with a short timeout, and without `shell=True`. For stronger isolation, judges can execute the same flow inside the provided Docker image using `docker run --network none esm-agentbench:ci` to disable external access.
