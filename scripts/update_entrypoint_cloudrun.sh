#!/usr/bin/env bash
# Deploy the assessor to Cloud Run and update agent_card.toml with the public URL.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/update_entrypoint_cloudrun.sh --project <gcp-project> [--region <region>] [--service <name>] [--force]

Deploys the repo to Cloud Run using gcloud, fetches the resulting HTTPS URL, updates agent_card.toml
so entrypoint points to /.well-known/agent-card.json, and commits the change.

Options:
  --project   GCP project ID (required)
  --region    Cloud Run region (default: us-central1)
  --service   Cloud Run service name (default: esm-agentbench)
  --force     Proceed even if git working tree has uncommitted changes
  -h, --help  Show this help message
USAGE
}

log() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
err() { >&2 log "ERROR: $*"; }

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command not found: $1"; exit 1
  fi
}

check_git_clean() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    err "This script must be run inside a git repository."; exit 1
  fi
  if [ "$force" -eq 0 ] && [ -n "$(git status --porcelain)" ]; then
    err "Working tree has uncommitted changes. Commit/stash them or rerun with --force."; exit 1
  fi
}

update_agent_card() {
  local entrypoint="$1"
  local path="agent_card.toml"
  if [ ! -f "$path" ]; then
    err "Missing $path in repo root."; exit 1
  fi
  log "Updating ${path} entrypoint to ${entrypoint}"
  ENTRYPOINT_VALUE="$entrypoint" python - <<'PY'
import os, pathlib, re
path = pathlib.Path('agent_card.toml')
text = path.read_text()
entry = os.environ['ENTRYPOINT_VALUE']
line = f'entrypoint = "{entry}"'
if re.search(r'^entrypoint\s*=.*$', text, flags=re.MULTILINE):
    text = re.sub(r'^entrypoint\s*=.*$', line, text, count=1, flags=re.MULTILINE)
else:
    m = re.search(r'^type\s*=.*$', text, flags=re.MULTILINE)
    if m:
        insert_at = m.end()
        text = text[:insert_at] + '\n' + line + text[insert_at:]
    else:
        if not text.endswith('\n'):
            text += '\n'
        text += line + '\n'
path.write_text(text)
PY
}

force=0
region="us-central1"
service="esm-agentbench"
project=""

while [ $# -gt 0 ]; do
  case "$1" in
    --project) project="$2"; shift;;
    --region) region="$2"; shift;;
    --service) service="$2"; shift;;
    --force) force=1;;
    -h|--help) usage; exit 0;;
    *) err "Unknown option: $1"; usage; exit 1;;
  esac
  shift
  done

if [ -z "$project" ]; then
  err "--project is required"; usage; exit 1
fi

require_command gcloud
require_command python
check_git_clean

log "Deploying to Cloud Run (service=${service}, region=${region}, project=${project})..."
gcloud run deploy "$service" --source . --region "$region" --project "$project" --quiet

log "Fetching service URL..."
url=$(gcloud run services describe "$service" --platform managed --region "$region" --project "$project" --format 'value(status.url)')
if [ -z "$url" ]; then
  err "Failed to obtain Cloud Run service URL"; exit 1
fi
log "Service URL: ${url}"
entrypoint="${url%/}/.well-known/agent-card.json"
update_agent_card "$entrypoint"

log "Creating git commit..."
git add agent_card.toml
# shellcheck disable=SC2016
commit_msg="CI: set agent_card entrypoint to Cloud Run URL (${service})"
git commit -m "$commit_msg"

log "Done. Updated entrypoint: ${entrypoint}"
