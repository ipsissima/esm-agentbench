# Spectral Certificates for Agent Drift Detection

## Mathematical Foundations: Wedin/Davis-Kahan Perturbation Bounds with Koopman Dynamics

**Authors:** ESM-AgentBench Team
**Date:** December 2025
**Status:** Formal Note

---

## Overview

This note establishes the mathematical foundations for spectral drift detection. The key theoretical results are:

1. **Wedin's Theorem (1972)**: Bounds perturbation of singular subspaces under matrix noise - this is the core theorem for our SVD-based certificates
2. **Davis-Kahan Theorem (1970)**: Bounds eigenspace perturbation for symmetric matrices (applies to Gram matrices)
3. **Koopman Operators**: Linear approximation of temporal dynamics for prediction residual computation

The detection pipeline uses **Wedin's theorem** to guarantee that subspace angles and reconstruction residuals reliably indicate trajectory perturbation.

---

## 1. Assumptions

We make the following assumptions for spectral certificate validity:

1. **Bounded Noise**: The embedding noise $\eta_t$ at each step satisfies $\|\eta_t\| \leq \eta_{\max}$ for some known bound $\eta_{\max}$.

2. **Local Koopman Linearizability**: Within each reasoning segment, the agent's state evolution can be locally approximated by a discrete-time linear operator:
   $$x_{t+1} \approx K x_t + \eta_t$$
   where $K \in \mathbb{R}^{d \times d}$ is the local Koopman operator approximation.

3. **Operator Norm Bounds**: The fitted Koopman operator satisfies $\|K\|_2 \leq \kappa_{\max}$ for numerical stability.

4. **Embedding Stability**: The embedding function $\phi: \text{text} \to \mathbb{R}^d$ has bounded Lipschitz constant $L_\phi$.

5. **Sufficient Data**: The trajectory has length $T \geq 2k$ for rank-$k$ approximation.

---

## 2. Definitions

### Trajectory Matrix
Given a sequence of embeddings $\{x_t\}_{t=1}^T$ where $x_t \in \mathbb{R}^d$, the **trajectory matrix** is:
$$X = \begin{bmatrix} x_1 & x_2 & \cdots & x_T \end{bmatrix} \in \mathbb{R}^{d \times T}$$

For affine dynamics handling, we augment with a bias row:
$$\tilde{X} = \begin{bmatrix} X \\ \mathbf{1}^T \end{bmatrix} \in \mathbb{R}^{(d+1) \times T}$$

### Singular Value Decomposition
The SVD of $\tilde{X}$ is:
$$\tilde{X} = U \Sigma V^T = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$
where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ are the singular values.

### Rank-k Best Approximation
The rank-$k$ best approximation is:
$$\tilde{X}_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$
where $U_k = [u_1, \ldots, u_k]$, $\Sigma_k = \text{diag}(\sigma_1, \ldots, \sigma_k)$, $V_k = [v_1, \ldots, v_k]$.

### Reconstruction Residual
The normalized reconstruction residual is:
$$\rho(X, k) = \frac{\|\tilde{X} - \tilde{X}_k\|_F}{\|\tilde{X}\|_F} = \sqrt{\frac{\sum_{i=k+1}^{r} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}}$$

### Spectral Gap
The **spectral gap** at rank $k$ is:
$$\delta_k = \sigma_k - \sigma_{k+1}$$
A larger gap indicates stronger separation between retained and discarded components.

### Tail Energy
The **tail energy** quantifies unexplained variance:
$$\tau_k = \frac{\sum_{i=k+1}^{r} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} = 1 - \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}$$

### Koopman Residual
Given projected states $Z = \tilde{X} V_k^T$, we fit the temporal operator:
$$A = \arg\min_{A} \|Z_1 - A Z_0\|_F$$
where $Z_0 = Z_{1:T-1}$ and $Z_1 = Z_{2:T}$.

The **Koopman residual** is:
$$r_K = \frac{\|Z_1 - A Z_0\|_F}{\|Z_1\|_F}$$

---

## 3. Wedin's Theorem (Singular Subspace Perturbation)

**Theorem (Wedin, 1972):** Let $X = U \Sigma V^T$ be the SVD of $X$, and let $\tilde{X} = X + E$ be a perturbed matrix with SVD $\tilde{X} = \tilde{U} \tilde{\Sigma} \tilde{V}^T$. Partition:
$$U = [U_1, U_2], \quad V = [V_1, V_2], \quad \Sigma = \begin{bmatrix} \Sigma_1 & 0 \\ 0 & \Sigma_2 \end{bmatrix}$$

If the spectral gap satisfies:
$$\delta = \min(\sigma_{\min}(\Sigma_1)) - \max(\sigma_{\max}(\Sigma_2)) > 0$$

Then the canonical angles $\Theta$ between the column spaces of $U_1$ and $\tilde{U}_1$ satisfy:
$$\|\sin \Theta(U_1, \tilde{U}_1)\|_F \leq \frac{\max(\|E^T U_1\|_F, \|E V_1\|_F)}{\delta}$$

**Conservative Bound:** Using the operator norm:
$$\sin \theta_{\max}(U_1, \tilde{U}_1) \leq \frac{\|E\|_2}{\delta}$$

---

## 4. Davis-Kahan Theorem (Eigenspace Perturbation)

**Theorem (Davis-Kahan, 1970):** For symmetric matrices $A$ and $\tilde{A} = A + E$, let $U_k$ and $\tilde{U}_k$ be the leading $k$ eigenvectors. If the eigenvalue gap $\delta_k = \lambda_k - \lambda_{k+1} > 0$, then:
$$\|\sin \Theta(U_k, \tilde{U}_k)\|_F \leq \frac{\|E\|_F}{\delta_k}$$

This applies to the Gram matrices $X^T X$ and their perturbations.

---

## 5. Detection Lemma

**Lemma (Spectral Drift Detection):** Let $X$ be a "clean" trajectory matrix (gold standard) and $\tilde{X} = X + E$ be a perturbed trajectory (drift or adversarial). Assume:
- Spectral gap: $\delta_k \geq \delta_0 > 0$
- Noise bound: $\|\eta\|_F \leq \eta_{\max}$
- Perturbation: $\|E\|_F \geq \varepsilon$

Then the following detection guarantees hold:

### (a) Subspace Angle Detection
The Davis-Kahan angle satisfies:
$$\theta_{\text{DK}} = \arcsin\left(\min\left(1, \frac{\|E\|_F}{\delta_k}\right)\right)$$

If $\|E\|_F \geq \varepsilon$ and $\delta_k \geq \delta_0$, then:
$$\theta_{\text{DK}} \geq \arcsin\left(\min\left(1, \frac{\varepsilon}{\delta_0 + \eta_{\max}}\right)\right)$$

### (b) Residual Detection
The reconstruction residual satisfies:
$$\rho(\tilde{X}, k) - \rho(X, k) \geq \gamma(\varepsilon, \delta_0, \eta_{\max})$$

where:
$$\gamma(\varepsilon, \delta_0, \eta_{\max}) = \frac{\varepsilon - \eta_{\max}}{\|X\|_F + \varepsilon} \cdot \frac{\delta_0}{\delta_0 + \sigma_1(X)}$$

### (c) Combined Certificate
The theoretical bound is:
$$B_{\text{theory}} = C_{\text{res}} \cdot r_K + C_{\text{tail}} \cdot \tau_k + C_{\text{sem}} \cdot d_{\text{sem}} + C_{\text{robust}} \cdot L_\phi$$

where constants $C_{\text{res}}, C_{\text{tail}}, C_{\text{sem}}, C_{\text{robust}} \leq 2.0$ are formally verified via Coq.

**Detection Rule:** Flag trajectory as drifted if:
$$B_{\text{theory}} > \tau \quad \text{OR} \quad \theta_{\text{DK}} > \theta_0$$

---

## 6. Proof Sketch

**Step 1 (Wedin's Theorem - Core Bound):** Apply Wedin's theorem to bound singular subspace perturbation:
$$\|\sin \Theta\| \leq \frac{\|E\|_2}{\sigma_k - \sigma_{k+1}}$$

This is the fundamental result: the subspace angle is controlled by the perturbation norm divided by the spectral gap.

**Step 2:** Relate the perturbation $E$ to drift:
- For adversarial drift: $E$ captures deviation from expected trajectory
- For noise: $E \sim \mathcal{N}(0, \sigma^2 I)$ with high-probability bounds

**Step 3:** The temporal prediction residual (Koopman-based) increases with perturbation:
$$\|Z_1^{\text{perturbed}} - A Z_0^{\text{perturbed}}\|_F \geq \|E_{1:} - A E_{0:}\|_F - \text{noise}$$

**Step 4:** By triangle inequality on the theoretical bound:
$$B_{\text{theory}}^{\text{perturbed}} - B_{\text{theory}}^{\text{clean}} \geq C_{\text{res}} \cdot \Delta r_K$$

where $\Delta r_K$ scales with $\|E\|_F / \|X\|_F$.

**Note:** The detection guarantee derives from **Wedin's theorem**, not from properties of the Koopman operator. The Koopman fit is used only to compute an auxiliary prediction residual feature.

---

## 7. Operational Rule: Threshold Selection

### Default Threshold
$$\tau = \bar{\rho}_{\text{baseline}} + 3 \cdot \sigma_{\text{baseline}}$$

where $\bar{\rho}_{\text{baseline}}$ and $\sigma_{\text{baseline}}$ are computed from gold-standard traces.

### Calibrated Threshold (ROC-based)
Given labeled data (gold, creative, drift), choose $\tau$ to achieve:
- **FPR** (False Positive Rate on creative) $\leq 0.05$
- Maximize **TPR** (True Positive Rate on drift)

The optimal threshold is found via:
```python
fpr, tpr, thresholds = roc_curve(y_true, scores)
idx = np.where(fpr <= 0.05)[0][-1]  # Last index where FPR <= 0.05
tau_calibrated = thresholds[idx]
```

### Rank Selection ($k$)

**Default:** $k = 10$ captures most variance in agent trajectories.

**Elbow Method:** Choose smallest $k$ such that:
$$\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \geq 0.90$$

**Adaptive:** Use cross-validation on gold traces to select $k$ minimizing baseline residual variance.

---

## 8. Implementation Notes

### Numerical Stability
1. **Normalize embeddings:** Zero-mean, unit variance per-run before SVD
2. **Regularization:** Add $\varepsilon I$ to Gram matrix for least-squares
3. **Clipping:** Bound angles to $[0, \pi/2]$

### Computational Complexity
- SVD: $O(d \cdot T \cdot \min(d, T))$
- Koopman fit: $O(k^3 + k^2 T)$
- Subspace angle: $O(k^3)$

---

## References

1. Davis, C., & Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. III. *SIAM Journal on Numerical Analysis*, 7(1), 1-46.

2. Wedin, P. Å. (1972). Perturbation bounds in connection with singular value decomposition. *BIT Numerical Mathematics*, 12(1), 99-111.

3. Mezić, I. (2013). Analysis of fluid flows via spectral properties of the Koopman operator. *Annual Review of Fluid Mechanics*, 45, 357-378.

4. Stewart, G. W., & Sun, J. (1990). *Matrix Perturbation Theory*. Academic Press.

5. ESM-AgentBench UELAT Formal Verification (2025). Coq proofs for spectral certificate bounds.

---

## Appendix: Real Agent Evaluation

**All benchmark evidence is derived from real agent traces.** Traces are generated by running actual HuggingFace-hosted language models (e.g., Qwen2.5-Coder, CodeLlama) on scenario tasks. The three trace categories are:

- **Gold:** Low-temperature (T≤0.3) agent behavior with deterministic tool use
- **Creative:** Higher-temperature (T≈0.7) exploration while remaining on-task
- **Drift:** Adversarially prompted or off-task trajectories

Synthetic trace generation code exists only in `legacy/` for historical reference and in `tests/` for numerical regression testing. CI guards prevent synthetic code from entering the evidence pipeline.

For real agent evaluation, see:
- `tools/real_agents_hf/run_real_agents.py` - Generate real agent traces
- `analysis/run_real_hf_experiment.py` - Analyze real agent traces
- `docs/REAL_AGENT_HF_EVAL.md` - Complete evaluation guide
