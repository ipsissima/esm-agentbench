# Benchmark Adapters

This directory contains adapters for running spectral certificate metrics against
public hallucination datasets.

## Running the real/primary adapter

`run_real_benchmark.py` prefers downloading the real validation splits from
Hugging Face. Pass `--use-sample-data` to run the bundled offline traces instead
(for CI or air-gapped environments).

```bash
# Evaluate against the real HaluEval validation split (default)
python benchmarks/run_real_benchmark.py

# Evaluate TruthfulQA
python benchmarks/run_real_benchmark.py --dataset truthfulqa

# Offline/CI mode using bundled traces
python benchmarks/run_real_benchmark.py --use-sample-data
```

Each run reports the AUC-ROC for detecting hallucinations (label `1`) using
`theoretical_bound` from `certificates.make_certificate.compute_certificate`, as
well as the average bound for good (label `0`) versus bad (label `1`) traces.

## Dataset loaders

All adapters delegate embedding and spectral bound computation to
`assessor.kickoff.embed_trace_steps` and
`certificates.make_certificate.compute_certificate`, enabling apples-to-apples
comparisons between spectral certificates and ground truth labels.
