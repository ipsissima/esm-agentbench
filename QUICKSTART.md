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
   - The script prints a JSON-formatted certificate with keys like `residual`, `tail_energy`, and `theoretical_bound`.
   - **Low bound ⇒ Coherent.** Reasoning stays within the SVD approximation; singular gap is stable.
   - **High bound ⇒ Drift.** The residual or tail energy explodes; flag for human review.
