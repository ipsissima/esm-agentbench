#!/usr/bin/env bash
# Generate uelat_constants.json from a Coq development.
#
# Usage:
#   tools/generate_uelat_constants_from_coq.sh path/to/file.v output.json C_tail C_res
#
# The script checks for `coqtop`, runs it in batch mode to `Print` each constant,
# captures the textual output, and converts it to JSON. Parsing is tolerant but
# if it fails, a diagnostic is shown along with a manual fallback command.
#
# Example:
#   tools/generate_uelat_constants_from_coq.sh ULELAT/analysis/constants_export.v \
#       certificates/uelat_constants.json C_tail C_res
#
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <coq-file.v> <output.json> <constant1> [constant2 ...]" >&2
  exit 1
fi

COQ_FILE=$1
OUTPUT_JSON=$2
shift 2
CONSTANTS=("$@")

if ! command -v coqtop >/dev/null 2>&1; then
  echo "Error: coqtop not found in PATH. Install Coq or load your opam switch." >&2
  echo "On Ubuntu: sudo apt-get install coq;" >&2
  exit 1
fi

if [[ ! -f "$COQ_FILE" ]]; then
  echo "Error: Coq file '$COQ_FILE' does not exist" >&2
  exit 1
fi

tmp_out=$(mktemp)
trap 'rm -f "$tmp_out"' EXIT

# Build eval flags for coqtop
EVAL_FLAGS=()
for c in "${CONSTANTS[@]}"; do
  EVAL_FLAGS+=( -eval "Print ${c}." )
done

set +e
coqtop -quiet -batch -l "$COQ_FILE" "${EVAL_FLAGS[@]}" > "$tmp_out" 2>&1
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "coqtop returned non-zero status ($status). Output:" >&2
  cat "$tmp_out" >&2
  exit $status
fi

python - "$OUTPUT_JSON" "$tmp_out" <<'PY'
import json
import re
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
coq_log = Path(sys.argv[2]).read_text(encoding="utf-8")
pattern = re.compile(r"(?P<name>[A-Za-z0-9_']+)\s*(?:[:=]|:=).*?(?P<val>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
constants = {}
for match in pattern.finditer(coq_log):
    name = match.group("name")
    val = float(match.group("val"))
    constants[name] = val

if not constants:
    sys.stderr.write("Failed to parse constants from coqtop output.\n")
    sys.stderr.write("Raw output follows:\n")
    sys.stderr.write(coq_log + "\n")
    sys.stderr.write(
        "Manual fallback: coqtop -batch -quiet -l <file.v> -eval 'Print C_tail.' "
        "> coq.txt && python - <<'PY'\\n"  # noqa: W605
        "import json,re\ntext=open('coq.txt').read()\n"
        "pat=re.compile(r\"(?P<name>[A-Za-z0-9_']+)\\s*[:=].*?(?P<val>[+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\")\n"
        "json.dump({m.group('name'): float(m.group('val')) for m in pat.finditer(text)}, open('uelat_constants.json','w'), indent=2)\n"
        "PY\n"
    )
    sys.exit(2)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(constants, indent=2), encoding="utf-8")
print(f"Wrote {len(constants)} constants to {out_path}")
PY

