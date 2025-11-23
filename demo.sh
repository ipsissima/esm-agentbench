#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PID_FILE="$ROOT_DIR/.demo_app.pid"
LOG_FILE="$ROOT_DIR/demo_app.log"

is_app_running() {
  curl -s -m 2 http://localhost:8080/.well-known/agent-card.json >/dev/null 2>&1
}

start_app() {
  if is_app_running; then
    echo "Assessor already running on port 8080"
    return
  fi
  echo "Starting assessor app in background..."
  python -m assessor.app >"$LOG_FILE" 2>&1 &
  echo $! >"$APP_PID_FILE"
  # Give it a moment to start
  for _ in {1..5}; do
    if is_app_running; then
      echo "Assessor started (pid $(cat "$APP_PID_FILE"))"
      return
    fi
    sleep 1
  done
  echo "Warning: assessor may not have started; check $LOG_FILE" >&2
}

start_app

echo "Running SWE demo episodes..."
python "$ROOT_DIR/tools/run_demo_swe.py"

REPORT_PATH="$ROOT_DIR/demo_swe/report.json"
echo "Demo report: $REPORT_PATH"
echo "Episode traces: $ROOT_DIR/demo_traces"
