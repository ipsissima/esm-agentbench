#!/usr/bin/env bash
# Update agent_card.toml entrypoint to point at an ngrok-exposed Flask assessor.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/update_entrypoint_ngrok.sh [--force] [--cleanup]

Starts the local assessor (unless SKIP_APP_START=1) and ngrok (using NGROK_PORT, default 8080),
waits for the ngrok public HTTPS URL, updates agent_card.toml entrypoint, and commits the change.

Options:
  --force    Proceed even if the git working tree has uncommitted changes.
  --cleanup  Kill any background processes started by this script before exiting.
USAGE
}

# Globals to remember background processes
app_pid=""
ngrok_pid=""
cleanup_on_exit=0
force=0

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

start_app() {
  if [ "${SKIP_APP_START:-0}" = "1" ]; then
    log "SKIP_APP_START=1 set; not starting Flask app."; return
  fi
  require_command python
  log "Starting Flask assessor (python assessor/app.py) in background..."
  python assessor/app.py >/tmp/assessor.log 2>&1 &
  app_pid=$!
  log "Assessor started with PID ${app_pid}; logs: /tmp/assessor.log"
}

start_ngrok() {
  require_command ngrok
  local port="${NGROK_PORT:-8080}"
  log "Starting ngrok on port ${port}..."
  ngrok http "${port}" >/tmp/ngrok.log 2>&1 &
  ngrok_pid=$!
  log "ngrok started with PID ${ngrok_pid}; logs: /tmp/ngrok.log"
}

wait_for_ngrok() {
  require_command curl
  local timeout=20
  local elapsed=0
  while [ ${elapsed} -lt ${timeout} ]; do
    if response=$(curl -sf http://127.0.0.1:4040/api/tunnels 2>/dev/null); then
      https_url=$(NGROK_API_RESP="$response" python - <<'PY'
import json, os, sys
resp = json.loads(os.environ.get('NGROK_API_RESP', '{}'))
for tunnel in resp.get('tunnels', []):
    url = tunnel.get('public_url', '')
    if url.startswith('https://'):
        print(url)
        sys.exit(0)
sys.exit(1)
PY
)
      if [ -n "${https_url:-}" ]; then
        log "ngrok public URL found: ${https_url}"
        return 0
      fi
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  err "Timed out waiting for ngrok API at http://127.0.0.1:4040/api/tunnels"
  exit 1
}

build_entrypoint() {
  local host="${https_url#https://}"
  entrypoint="https://${host}/.well-known/agent-card.json"
}

update_toml() {
  local path="agent_card.toml"
  if [ ! -f "$path" ]; then
    err "Missing $path in repo root."; exit 1
  fi
  log "Updating ${path} entrypoint to ${entrypoint}"
  ENTRYPOINT_VALUE="$entrypoint" python - <<'PY'
import pathlib, re, os
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

commit_changes() {
  log "Creating git commit..."
  git add agent_card.toml
  git commit -m "CI: set agent_card entrypoint to ngrok URL"
}

cleanup() {
  if [ "$cleanup_on_exit" -eq 0 ]; then
    return
  fi
  if [ -n "$ngrok_pid" ] && kill -0 "$ngrok_pid" >/dev/null 2>&1; then
    log "Stopping ngrok (PID ${ngrok_pid})"; kill "$ngrok_pid" || true
  fi
  if [ -n "$app_pid" ] && kill -0 "$app_pid" >/dev/null 2>&1; then
    log "Stopping assessor (PID ${app_pid})"; kill "$app_pid" || true
  fi
}
trap cleanup EXIT

while [ $# -gt 0 ]; do
  case "$1" in
    --force) force=1;;
    --cleanup) cleanup_on_exit=1;;
    -h|--help) usage; exit 0;;
    *) err "Unknown option: $1"; usage; exit 1;;
  esac
  shift
done

check_git_clean
start_app
start_ngrok
wait_for_ngrok
build_entrypoint
update_toml
commit_changes

log "Done. Public URL: ${https_url}"
log "Updated: $(pwd)/agent_card.toml"
if [ $cleanup_on_exit -eq 0 ]; then
  log "Processes left running: app PID=${app_pid:-N/A}, ngrok PID=${ngrok_pid:-N/A}"
else
  log "Cleanup requested; background processes terminated."
fi
