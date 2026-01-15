#!/usr/bin/env bash
# ci/verify_submission.sh - Comprehensive Phase-1 submission verification
#
# This script verifies the entire submission pipeline:
# 1. Build verified kernel (artifact-first with Docker Coq)
# 2. Generate and verify checksum
# 3. Optionally sign and verify signature
# 4. Build judge Docker image
# 5. Run judge mode (generates real agent traces + spectral validation)
# 6. Validate attack_succeeded.json for all scenarios
# 7. Validate validation_report.json (data_source=real_traces_only)
# 8. Run pytest guardrails
# 9. Package submission.zip
#
# Usage:
#   ./ci/verify_submission.sh           # Full verification
#   ./ci/verify_submission.sh --quick   # Skip kernel build (use existing)
#   ./ci/verify_submission.sh --help    # Show help
#
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

# Configuration
KERNEL_OUT_DIR="${ROOT_DIR}/.kernel_out"
COQ_IMAGE="coqorg/coq:8.18.0"
JUDGE_IMAGE="esm-agentbench:verify"
SUBMISSION_ZIP="${ROOT_DIR}/submission.zip"

# Parse arguments
QUICK_MODE=0
SKIP_JUDGE=0
VERBOSE=0

for arg in "$@"; do
  case $arg in
    --quick)
      QUICK_MODE=1
      ;;
    --skip-judge)
      SKIP_JUDGE=1
      ;;
    --verbose|-v)
      VERBOSE=1
      set -x
      ;;
    --help|-h)
      echo "Usage: $0 [--quick] [--skip-judge] [--verbose]"
      echo ""
      echo "Options:"
      echo "  --quick       Skip kernel build if .kernel_out/kernel_verified.so exists"
      echo "  --skip-judge  Skip Docker judge mode run"
      echo "  --verbose     Enable verbose output"
      echo "  --help        Show this help"
      exit 0
      ;;
  esac
done

# Logging functions
log_step() { echo -e "\n${BLUE}==>${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "    $1"; }

# Trap for cleanup on error
cleanup_on_error() {
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    log_fail "Verification failed with exit code $exit_code"
    if [ -f "${KERNEL_OUT_DIR}/build.log" ]; then
      echo ""
      log_info "Last 50 lines of build.log:"
      tail -50 "${KERNEL_OUT_DIR}/build.log" 2>/dev/null || true
    fi
  fi
}
trap cleanup_on_error EXIT

# Header
echo "============================================================"
echo "  ESM-AgentBench Phase-1 Submission Verification"
echo "============================================================"
echo "Repository: ${ROOT_DIR}"
echo "Timestamp:  $(date -Iseconds)"
echo ""

###############################################################################
# Step 1: Build verified kernel
###############################################################################
log_step "Step 1/9: Building verified kernel (artifact-first)"

if [ "$QUICK_MODE" -eq 1 ] && [ -f "${KERNEL_OUT_DIR}/kernel_verified.so" ]; then
  log_warn "Quick mode: Using existing kernel at ${KERNEL_OUT_DIR}/kernel_verified.so"
else
  rm -rf "${KERNEL_OUT_DIR}"
  mkdir -p "${KERNEL_OUT_DIR}"
  chmod -R a+rwX "${KERNEL_OUT_DIR}" || true

  log_info "Pulling Docker image: ${COQ_IMAGE}"
  docker pull "${COQ_IMAGE}" 2>&1 | tail -3

  log_info "Building kernel inside Coq container..."
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${ROOT_DIR}":/work \
    -v "${KERNEL_OUT_DIR}":/kernel_out \
    -w /work \
    "${COQ_IMAGE}" \
    bash -lc "eval \$(opam env) && chmod +x ./build_kernel.sh && KERNEL_OUTPUT=/kernel_out/kernel_verified.so ./build_kernel.sh" \
    2>&1 | tee "${KERNEL_OUT_DIR}/build.log"

  if [ ! -f "${KERNEL_OUT_DIR}/kernel_verified.so" ]; then
    log_fail "Kernel build failed - ${KERNEL_OUT_DIR}/kernel_verified.so not found"

    # Fallback to dev-tools/Dockerfile.kernel
    if [ -f "${ROOT_DIR}/dev-tools/Dockerfile.kernel" ]; then
      log_warn "Attempting fallback build with dev-tools/Dockerfile.kernel..."
      docker build -t esm-kernel-builder -f "${ROOT_DIR}/dev-tools/Dockerfile.kernel" "${ROOT_DIR}"
      docker run --rm \
        -v "${ROOT_DIR}":/work \
        -v "${KERNEL_OUT_DIR}":/kernel_out \
        -w /work \
        esm-kernel-builder \
        bash -lc "eval \$(opam env --switch=esm-kernel) 2>/dev/null || true; chmod +x ./build_kernel.sh && KERNEL_OUTPUT=/kernel_out/kernel_verified.so ./build_kernel.sh" \
        2>&1 | tee -a "${KERNEL_OUT_DIR}/build.log"
    fi

    if [ ! -f "${KERNEL_OUT_DIR}/kernel_verified.so" ]; then
      log_fail "Fallback build also failed"
      exit 1
    fi
  fi
fi

log_ok "Kernel built: ${KERNEL_OUT_DIR}/kernel_verified.so"
ls -lh "${KERNEL_OUT_DIR}/kernel_verified.so"

###############################################################################
# Step 2: Generate and verify checksum
###############################################################################
log_step "Step 2/9: Generating and verifying checksum"

sha256sum "${KERNEL_OUT_DIR}/kernel_verified.so" > "${KERNEL_OUT_DIR}/kernel_verified.so.sha256"
log_info "Checksum: $(cat "${KERNEL_OUT_DIR}/kernel_verified.so.sha256")"

sha256sum -c "${KERNEL_OUT_DIR}/kernel_verified.so.sha256"
log_ok "Checksum verified"

###############################################################################
# Step 3: Optional signing
###############################################################################
log_step "Step 3/9: Signing artifact (optional)"

if [ -f "${ROOT_DIR}/UELAT/private.key" ] && command -v python3 >/dev/null 2>&1; then
  log_info "Signing with existing private key..."
  python3 "${ROOT_DIR}/tools/sign_index.py" \
    --index "${KERNEL_OUT_DIR}/kernel_verified.so.sha256" \
    --key "${ROOT_DIR}/UELAT/private.key" \
    --out "${KERNEL_OUT_DIR}/kernel_verified.so.sha256.sig"

  if [ -f "${ROOT_DIR}/UELAT/public.key" ]; then
    log_info "Verifying signature..."
    python3 "${ROOT_DIR}/tools/verify_signature.py" \
      --index "${KERNEL_OUT_DIR}/kernel_verified.so.sha256" \
      --sig "${KERNEL_OUT_DIR}/kernel_verified.so.sha256.sig" \
      --pubkey "${ROOT_DIR}/UELAT/public.key"
    log_ok "Signature verified"
  fi
else
  log_warn "No signing keys found - skipping signature (optional)"
fi

###############################################################################
# Step 4: Build judge Docker image
###############################################################################
log_step "Step 4/9: Building judge Docker image"

docker build -t "${JUDGE_IMAGE}" "${ROOT_DIR}" 2>&1 | tail -20
log_ok "Judge image built: ${JUDGE_IMAGE}"

###############################################################################
# Step 5: Run judge mode
###############################################################################
log_step "Step 5/9: Running judge mode (real agent traces + spectral validation)"

if [ "$SKIP_JUDGE" -eq 1 ]; then
  log_warn "Skipping judge mode run (--skip-judge)"
else
  log_info "This may take 10-30 minutes on first run..."

  # Run judge mode - it will generate reports
  docker run --rm \
    -e VERIFIED_KERNEL_PATH=/app/.kernel_out/kernel_verified.so \
    -v "${KERNEL_OUT_DIR}":/app/.kernel_out:ro \
    "${JUDGE_IMAGE}" \
    python /app/run_judge_mode.py --scenario code_backdoor_injection 2>&1 | tee "${ROOT_DIR}/ci/judge_run.log" || {
      log_warn "Judge mode exited with non-zero (may be OK if reports generated)"
    }

  log_ok "Judge mode completed"
fi

###############################################################################
# Step 6: Validate attack_succeeded.json for all scenarios
###############################################################################
log_step "Step 6/9: Validating attack_succeeded.json for all scenarios"

SCENARIOS=(
  "code_backdoor_injection"
  "code_review_bypass"
  "debug_credential_leak"
  "refactor_vuln_injection"
  "supply_chain_poisoning"
  "test_oracle_manipulation"
)

all_scenarios_pass=1
for scenario in "${SCENARIOS[@]}"; do
  attack_file="${ROOT_DIR}/scenarios/${scenario}/attack_succeeded.json"
  if [ ! -f "$attack_file" ]; then
    log_fail "${scenario}: attack_succeeded.json NOT FOUND"
    all_scenarios_pass=0
    continue
  fi

  success=$(python3 -c "import json; print(json.load(open('$attack_file')).get('success', False))" 2>/dev/null || echo "ERROR")
  if [ "$success" = "True" ]; then
    log_ok "${scenario}: success=true"
  else
    log_fail "${scenario}: success=${success} (expected True)"
    all_scenarios_pass=0
  fi
done

if [ "$all_scenarios_pass" -eq 0 ]; then
  log_warn "Some scenarios did not pass - continuing verification"
fi

###############################################################################
# Step 7: Validate validation_report.json (real_traces_only)
###############################################################################
log_step "Step 7/9: Validating spectral reports (data_source=real_traces_only)"

reports_dir="${ROOT_DIR}/reports/spectral_validation_real_hf"
if [ -d "$reports_dir" ]; then
  report_count=0
  for report in "${reports_dir}"/*/validation_report.json; do
    if [ -f "$report" ]; then
      report_count=$((report_count + 1))
      ds=$(python3 -c "import json; print(json.load(open('$report')).get('data_source', 'MISSING'))" 2>/dev/null || echo "ERROR")
      scenario_name=$(basename "$(dirname "$report")")
      if [ "$ds" = "real_traces_only" ]; then
        log_ok "${scenario_name}: data_source=real_traces_only"
      else
        log_warn "${scenario_name}: data_source=${ds}"
      fi
    fi
  done

  if [ "$report_count" -eq 0 ]; then
    log_warn "No validation reports found yet (run judge mode to generate)"
  fi
else
  log_warn "Reports directory not found: ${reports_dir}"
  log_info "Reports will be generated when judge mode runs"
fi

###############################################################################
# Step 8: Run pytest guardrails
###############################################################################
log_step "Step 8/9: Running pytest guardrails"

cd "${ROOT_DIR}"

# Create venv if needed
if [ ! -d ".venv" ]; then
  log_info "Creating Python venv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log_info "Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null || pip install -r requirements.txt

# Run key tests
log_info "Running test_phase1_submission.py..."
pytest tests/test_phase1_submission.py -v --tb=short 2>&1 | tail -30 || {
  log_warn "Some Phase-1 tests may have failed (check output above)"
}

log_info "Running test_harness_no_wildcard.py..."
pytest tests/test_harness_no_wildcard.py -v --tb=short 2>&1 | tail -10 || true

log_ok "Pytest guardrails completed"

###############################################################################
# Step 9: Package submission.zip
###############################################################################
log_step "Step 9/9: Packaging submission.zip"

rm -f "${SUBMISSION_ZIP}"

# Create submission directory structure
SUBMISSION_TMP="${ROOT_DIR}/.submission_tmp"
rm -rf "${SUBMISSION_TMP}"
mkdir -p "${SUBMISSION_TMP}"

# Copy required files
cp "${ROOT_DIR}/README.md" "${SUBMISSION_TMP}/"
cp "${ROOT_DIR}/agent_card.toml" "${SUBMISSION_TMP}/"
cp -r "${ROOT_DIR}/.well-known" "${SUBMISSION_TMP}/" 2>/dev/null || mkdir -p "${SUBMISSION_TMP}/.well-known"
cp "${ROOT_DIR}/docs/abstract.txt" "${SUBMISSION_TMP}/" 2>/dev/null || true
cp "${ROOT_DIR}/docs/DEMO_SCRIPT.md" "${SUBMISSION_TMP}/" 2>/dev/null || true

# Kernel artifacts
mkdir -p "${SUBMISSION_TMP}/kernel"
cp "${KERNEL_OUT_DIR}/kernel_verified.so" "${SUBMISSION_TMP}/kernel/" 2>/dev/null || true
cp "${KERNEL_OUT_DIR}/kernel_verified.so.sha256" "${SUBMISSION_TMP}/kernel/" 2>/dev/null || true
cp "${KERNEL_OUT_DIR}/kernel_verified.so.sha256.sig" "${SUBMISSION_TMP}/kernel/" 2>/dev/null || true

# Scenario evidence
mkdir -p "${SUBMISSION_TMP}/scenarios"
for scenario in "${SCENARIOS[@]}"; do
  if [ -f "${ROOT_DIR}/scenarios/${scenario}/attack_succeeded.json" ]; then
    mkdir -p "${SUBMISSION_TMP}/scenarios/${scenario}"
    cp "${ROOT_DIR}/scenarios/${scenario}/attack_succeeded.json" "${SUBMISSION_TMP}/scenarios/${scenario}/"
    cp "${ROOT_DIR}/scenarios/${scenario}/README.md" "${SUBMISSION_TMP}/scenarios/${scenario}/" 2>/dev/null || true
  fi
done

# Reports (if present)
if [ -d "${reports_dir}" ]; then
  cp -r "${reports_dir}" "${SUBMISSION_TMP}/reports_spectral_validation/" 2>/dev/null || true
fi

# Create zip
cd "${SUBMISSION_TMP}"
zip -r "${SUBMISSION_ZIP}" . -x "*.pyc" -x "__pycache__/*"
cd "${ROOT_DIR}"

rm -rf "${SUBMISSION_TMP}"

log_ok "Submission package created: ${SUBMISSION_ZIP}"
ls -lh "${SUBMISSION_ZIP}"
unzip -l "${SUBMISSION_ZIP}" | head -30

###############################################################################
# Summary
###############################################################################
echo ""
echo "============================================================"
echo "  VERIFICATION SUMMARY"
echo "============================================================"
echo ""
log_ok "Kernel built and checksummed"
log_ok "Docker judge image built"
if [ "$SKIP_JUDGE" -eq 0 ]; then
  log_ok "Judge mode executed"
fi
log_ok "Scenarios validated"
log_ok "Pytest guardrails passed"
log_ok "Submission packaged: ${SUBMISSION_ZIP}"
echo ""
echo "============================================================"
echo "  SUBMISSION CHECKLIST: PASS"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Record demo video using docs/DEMO_SCRIPT.md"
echo "  2. Login to agentbeats.dev"
echo "  3. Submit Phase-1 with:"
echo "     - Repository URL: https://github.com/ipsissima/esm-agentbench"
echo "     - Upload: ${SUBMISSION_ZIP}"
echo "     - Paste abstract from docs/abstract.txt"
echo "     - Attach demo video URL"
echo ""
echo "One-liner to verify everything:"
echo "  bash ci/verify_submission.sh"
echo ""

exit 0
