#!/usr/bin/env python3
# scripts/update_agent_card.py
"""
Update agent_card.toml `entrypoint = "..."` with the given URL, and write .well-known/agent-card.json.
Idempotent and safe: it will replace an existing entrypoint line or add one at the top.
"""
import argparse
import json
import os
import re
from pathlib import Path
import tomllib  # reading only (Python 3.11+ or tomli fallback). For writing we will dump JSON.

def replace_entrypoint_in_toml(path: Path, url: str) -> None:
    text = path.read_text(encoding="utf-8")
    # Replace a line like: entrypoint = "..." (preserve indentation)
    if re.search(r'^\s*entrypoint\s*=\s*".*"', text, flags=re.MULTILINE):
        text_new = re.sub(r'^\s*entrypoint\s*=\s*".*"', f'entrypoint = "{url}"', text, flags=re.MULTILINE)
    else:
        # No entrypoint field found; add at top
        text_new = f'entrypoint = "{url}"\n' + text
    path.write_text(text_new, encoding="utf-8")

def write_agent_card_json_from_toml(toml_path: Path, json_path: Path) -> None:
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Cloud Run service URL, e.g. https://my-service-xyz.a.run.app")
    p.add_argument("--toml", default="agent_card.toml")
    p.add_argument("--json_out", default=".well-known/agent-card.json")
    args = p.parse_args()

    toml_path = Path(args.toml)
    if not toml_path.exists():
        raise SystemExit(f"agent_card.toml not found at {toml_path.resolve()}")

    replace_entrypoint_in_toml(toml_path, args.url)
    write_agent_card_json_from_toml(toml_path, Path(args.json_out))
    print(f"Updated {toml_path} and wrote {args.json_out}")

if __name__ == "__main__":
    main()
