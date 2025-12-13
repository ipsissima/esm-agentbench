# Theory: Certified AI Reasoning via Spectral Certificates

## Part 1: The Intuition (Accessible)
- **Koopman Operator Theory = linearizing the nonlinear reasoning path.** We lift token-level trajectories into an observable space where the Koopman operator acts linearly, letting us study stability with spectra rather than brittle prompts.
- **Why LLM-as-a-judge fails:** It is subjective, prompt-sensitive, and cannot bound risk. Two prompts may yield different "judges" and no guarantee is offered when hallucination creeps in.
- **Why Spectral Certificates win:** They measure how well a linear Koopman proxy explains the trajectory. A small theoretical bound means the proxy faithfully captures reasoning; spikes flag drift or hallucination before they surface externally.

## Part 2: The Math (Rigorous)
The definitions match the formal appendix and UELAT setup.

### Setup from the appendix
- Embeddings: a sequence of vectors $x_t \in \mathbb{R}^d$ for $t = 0, \ldots, T-1$.
- Affine drift handling: augment with bias $1$ to get $\tilde{x}_t = [x_t; 1]$.
- Finite-rank proxy: project onto $U \in \mathbb{R}^{(d+1) \times r}$ with orthonormal columns. The retained variance fraction is $\text{pca\_explained} = \sum_{i=1}^r \lambda_i / \sum_j \lambda_j$ with PCA eigenvalues $\lambda_i$.
- Koopman approximation: in reduced space $z_t = U^\top \tilde{x}_t$, fit a linear map $A$ minimizing $\lVert z_{t+1} - A z_t \rVert_F^2$.

### Koopman operator
Let $g$ be an observable on the state space. The Koopman operator $K$ acts as $(K g)(x) = g(f(x))$ for underlying dynamics $x_{t+1} = f(x_t)$. In the finite-rank approximation we identify $K$ with the fitted matrix $A$ over reduced observables.

### Residual
\[
\text{residual} = \frac{\|Z_1 - A Z_0\|_F}{\|Z_1\|_F + \epsilon}, \quad Z_0 = [z_0, \ldots, z_{T-2}], \; Z_1 = [z_1, \ldots, z_{T-1}].
\]

### Theoretical bound
\[
\text{bound} = \text{residual} + (1 - \text{pca\_explained}).
\]
This matches $\text{theoretical\_bound} = \text{residual} + \text{pca\_tail\_estimate}$ in the appendix, providing a conservative reconstruction error bound.

### Spectral stability (Davis–Kahan)
The Davis–Kahan theorem guarantees that if the spectral gap of $A$ is separated, the principal eigenspaces of the Koopman proxy remain stable under perturbations. Therefore, small residuals imply the spectral certificate’s leading directions—and thus the reasoning trajectory—stay consistent.
