# Theory: Spectral Certificates via SVD and Wedin's Theorem

## Part 1: The Intuition (Accessible)

- **Koopman Operator Theory = linearizing the nonlinear reasoning path.** We lift token-level trajectories into an observable space where a linear operator approximates the dynamics, enabling stability analysis via well-understood linear algebra.

- **Why LLM-as-a-judge fails:** It is subjective, prompt-sensitive, and cannot bound risk. Two prompts may yield different "judges" and no formal guarantee is offered.

- **Why Spectral Certificates help:** They measure how well a linear Koopman proxy explains the trajectory. A small theoretical bound means the proxy faithfully captures the dynamics; spikes indicate the model is behaving inconsistently.

## Part 2: The Math (Rigorous)

This section describes the **Constructive Finite-Rank Approximation** used in the certificate computation. The guarantees are **internal to the finite-rank model** and characterize how well the Koopman proxy fits the observed trajectory.

### Setup

- **Embeddings:** A sequence of vectors $x_t \in \mathbb{R}^d$ for $t = 0, \ldots, T-1$.
- **Affine augmentation:** Append bias $1$ to get $\tilde{x}_t = [x_t; 1]$, handling affine drift.
- **Finite-rank proxy:** Use Singular Value Decomposition to project onto the top $r$ right singular vectors.

### Singular Value Decomposition

Given the data matrix $X \in \mathbb{R}^{T \times (d+1)}$ (augmented embeddings), compute:

$$X = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{T \times T}$ contains left singular vectors
- $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{\min(T, d+1)})$ with $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{(d+1) \times (d+1)}$ contains right singular vectors

**Truncation:** Retain only the top $r$ components:
- $V_r$: first $r$ right singular vectors
- Projection: $z_t = V_r^T \tilde{x}_t \in \mathbb{R}^r$

**Energy decomposition:**
- Retained energy: $E_r = \sum_{i=1}^r \sigma_i^2$
- Total energy: $E = \sum_i \sigma_i^2$
- **Tail energy:** $\text{tail\_energy} = 1 - E_r / E$ (fraction not captured by truncation)

### Koopman Operator Approximation

In the reduced space, fit a linear map $A \in \mathbb{R}^{r \times r}$ minimizing:

$$\min_A \|Z_1 - A Z_0\|_F^2$$

where $Z_0 = [z_0, \ldots, z_{T-2}]$ and $Z_1 = [z_1, \ldots, z_{T-1}]$.

**Residual:**
$$\text{residual} = \frac{\|Z_1 - A Z_0\|_F}{\|Z_1\|_F + \epsilon}$$

This measures the normalized prediction error of the Koopman approximation.

### Theoretical Bound

The certificate computes a rigorous bound on reconstruction error:

$$\text{theoretical\_bound} = C_{\text{res}} \cdot \text{residual} + C_{\text{tail}} \cdot \text{tail\_energy}$$

where $C_{\text{res}}$ and $C_{\text{tail}}$ are constants from formal verification (see `UELAT/spectral_bounds.v`).

**Interpretation:**
- **Residual term:** Error from the Koopman approximation within the truncated subspace
- **Tail term:** Error from projecting onto a finite-rank subspace (information loss)

### Spectral Stability via Wedin's Theorem

**Why we use SVD instead of eigenvalues:** The Koopman operator $A$ is generally non-symmetric. Eigenvalue analysis of non-symmetric matrices is numerically unstable under perturbation. Singular values, however, are stable.

**Wedin's Theorem (1972):** Let $A, \tilde{A} \in \mathbb{R}^{m \times n}$ with SVDs $A = U \Sigma V^T$ and $\tilde{A} = \tilde{U} \tilde{\Sigma} \tilde{V}^T$. If the singular value gap $\delta = \sigma_r - \sigma_{r+1} > 0$, then the angle $\theta$ between the left/right singular subspaces satisfies:

$$\sin(\theta) \leq \frac{\max(\|E\|_F, \|E^T\|_F)}{\delta}$$

where $E = \tilde{A} - A$ is the perturbation.

**Application:** This theorem guarantees that:
1. Small perturbations to the trajectory produce small changes in the singular subspace
2. The Koopman operator's singular values characterize its conditioning
3. The spectral gap indicates robustness: large gaps mean stable subspace identification

### Singular Value Metrics

The certificate reports:
- **sigma_max, sigma_second:** Leading singular values of the data matrix
- **singular_gap:** $\sigma_1 - \sigma_2$ (stability indicator)
- **koopman_sigma_max:** Largest singular value of $A$ (operator norm)
- **koopman_singular_gap:** Gap in Koopman operator singular values

## Part 3: What This Does NOT Claim

**Scope of guarantees:** The spectral certificate characterizes the **finite-rank approximation quality**, not:
- The "correctness" of the LLM's reasoning
- Whether the output is factually accurate
- Any property of the LLM's internal mechanisms

**Finite-rank limitation:** We approximate an infinite-dimensional Koopman operator with a rank-$r$ matrix. The tail energy quantifies information lost to truncation but cannot recover it.

**Observability assumption:** We only observe embeddings, not internal states. The certificate measures trajectory consistency in embedding space.

## References

1. Wedin, P.Å. (1972). "Perturbation bounds in connection with singular value decomposition." BIT Numerical Mathematics, 12(1), 99-111.

2. Koopman, B.O. (1931). "Hamiltonian systems and transformation in Hilbert space." Proceedings of the National Academy of Sciences, 17(5), 315-318.

3. Stewart, G.W. & Sun, J.G. (1990). "Matrix Perturbation Theory." Academic Press.
