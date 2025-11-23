#!/usr/bin/env bash
# scripts/make_local_agent_card.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_CARD="$ROOT/agent_card.toml"
LOCAL_URL="http://localhost:8080/.well-known/agent-card.json"

if [ ! -f "$AGENT_CARD" ]; then
  echo "agent_card.toml not found at $AGENT_CARD" >&2
  exit 1
fi

python3 "$ROOT/scripts/update_agent_card.py" --url "$LOCAL_URL"
echo "Set agent_card entrypoint to $LOCAL_URL"
