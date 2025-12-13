# Benchmark Adapters

This directory contains lightweight adapters for running spectral certificate
metrics against public hallucination datasets. The provided scripts use mock
data to keep them runnable offline; swap in real loaders to evaluate on
HaluEval or TruthfulQA.

## Running the mock demos

```bash
python benchmarks/run_halueval.py
python benchmarks/run_truthfulqa.py
```

Each script prints the AUC-ROC for detecting hallucinations (label `1`) using
`theoretical_bound` from `certificates.make_certificate.compute_certificate`, as
well as the average bound for good (label `0`) versus bad (label `1`) traces.

## Running real datasets

The `benchmarks/run_real_benchmark.py` adapter loads actual validation splits
from Hugging Face (if the `datasets` library is installed) and measures how well
the spectral certificate separates truthful and hallucinated answers.

```bash
python benchmarks/run_real_benchmark.py --dataset halueval
python benchmarks/run_real_benchmark.py --dataset truthfulqa
```

If `datasets` is not installed, the script will print a helpful message and
exit without failing the workflow, keeping CI green even in minimal
environments.

## Using real datasets

Replace the mock loaders with Hugging Face datasets. Example skeleton:

```python
from datasets import load_dataset

# HaluEval
raw = load_dataset("halueval")
# TruthfulQA (via alpaca-style format)
raw = load_dataset("tatsu-lab/alpaca_eval")

samples = []
for row in raw["train"]:  # pick the split you want
    samples.append({
        "trace": [
            {"text": f"Question: {row['question']}"},
            {"text": f"Answer: {row['answer']}"},
        ],
        "label": int(row.get("label", 0)),  # 1 = hallucination, 0 = truthful
    })

results = evaluate_dataset(samples)
print_metrics(results)
```

Make sure to install dependencies first:

```bash
pip install datasets scikit-learn
```

These adapters are intentionally minimal: they delegate embedding and spectral
bound computation to `assessor.kickoff.embed_trace_steps` and
`certificates.make_certificate.compute_certificate`, enabling apples-to-apples
comparisons between spectral certificates and ground truth labels.
