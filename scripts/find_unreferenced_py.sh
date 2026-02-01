#!/usr/bin/env bash
set -euo pipefail

echo "Scanning for .py files without import references (heuristic)."

ROOT="$(git rev-parse --show-toplevel)"
TMP_LIST="$(mktemp)"
trap 'rm -f "${TMP_LIST}"' EXIT

# List all tracked .py files excluding tests, archive, and kernel prototypes.
git ls-files '*.py' \
  | grep -v '^tests/' \
  | grep -v '^archive/' \
  | grep -v '^kernel/prototype/' \
  > "${TMP_LIST}"

while read -r file; do
  base="$(basename "${file}" .py)"
  matches="$(
    git grep -n --hidden -E "(^|[.(])${base}($|[).])" -- ':!tests/*' ':!archive/*' \
      | wc -l \
      || true
  )"

  if [ "${matches}" -eq 0 ]; then
    echo "POSSIBLE UNUSED: ${file}"
  fi
done < "${TMP_LIST}"
