#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

if [ -z "${VERIFIED_KERNEL_PATH:-}" ]; then
  if [ "${SKIP_KERNEL_BUILD:-0}" = "1" ]; then
    echo "Skipping verified kernel build (SKIP_KERNEL_BUILD=1)."
  else
    echo "VERIFIED_KERNEL_PATH not set; building verified kernel..."
    ./build_kernel.sh || echo "Verified kernel build failed; continuing with fallback."
  fi
fi

KERNEL_SOURCE="${ROOT_DIR}/UELAT/kernel_verified.so"
FINAL_ARTIFACT_DIR="${ROOT_DIR}/reports/final_artifact"
if [ -f "${KERNEL_SOURCE}" ]; then
  mkdir -p "${FINAL_ARTIFACT_DIR}"
  cp "${KERNEL_SOURCE}" "${FINAL_ARTIFACT_DIR}/"
  sha256sum "${KERNEL_SOURCE}" | tee "${FINAL_ARTIFACT_DIR}/kernel_verified.sha256"
  echo "Included kernel in final artifact: ${FINAL_ARTIFACT_DIR}/kernel_verified.so"
fi

python tools/tune_metric.py --features-csv reports/features_dev.csv --seed 0
python tools/eval_holdout.py --model reports/best_model.pkl --features-csv reports/features_holdout.csv --fpr-target 0.05 --n-boot 1000 --seed 0

TRACES_DIR="${TRACES_DIR:-submissions/ipsissima}"
if [ ! -d "$TRACES_DIR" ]; then
  echo "Traces directory $TRACES_DIR not found; falling back to experiment_traces"
  TRACES_DIR="experiment_traces"
fi

python analysis/run_experiment.py --all-scenarios --traces-dir "$TRACES_DIR" --use-augmented-classifier --fpr-target 0.05

python tools/adversarial_test.py --model reports/best_model.pkl --traces-dir "$TRACES_DIR" --attack finite-diff --budget 50 --seed 0 --norm-clip 0.1

python - <<'PY'
import json
from pathlib import Path

reports = Path("reports") / "spectral_validation"
missing = []
for report in reports.glob("*/validation_report.json"):
    data = json.loads(report.read_text())
    if "data_source" not in data:
        missing.append(f"{report}: missing data_source")
    augmented = data.get("augmented_classifier")
    if augmented:
        if "auc_ci" not in augmented or "tpr_ci" not in augmented:
            missing.append(f"{report}: missing bootstrap CI fields")

if missing:
    raise SystemExit("\n".join(missing))
print("validation_report.json includes data_source and bootstrap CI fields")
PY

if [ -n "${SIGN_INDEX_WITH:-}" ]; then
  echo "SIGN_INDEX_WITH set; signing index.json files."
  while IFS= read -r -d '' index_file; do
    python tools/sign_index.py --index "$index_file" --key "$SIGN_INDEX_WITH" --out "${index_file}.sig"
  done < <(find "$TRACES_DIR" -name index.json -print0)
fi

echo "Final artifact production completed."
