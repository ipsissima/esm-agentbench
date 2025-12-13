# Quickstart (5 minutes)

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the basic example**
   ```bash
   python examples/basic_certificate.py
   ```

3. **Interpret the output**
   - The script prints a JSON-formatted certificate with keys like `residual`, `pca_tail_estimate`, and `theoretical_bound`.
   - **Low bound ⇒ Coherent.** Reasoning stays within the Koopman approximation; spectral gap is stable.
   - **High bound ⇒ Drift.** The residual or PCA tail explodes; flag for human review.
