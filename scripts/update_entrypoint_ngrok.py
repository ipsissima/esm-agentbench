#!/usr/bin/env python
"""Update agent_card.toml entrypoint to the current ngrok HTTPS URL."""

import argparse
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Optional

requests_spec = importlib.util.find_spec("requests")
if requests_spec:
    requests = importlib.import_module("requests")
else:  # pragma: no cover
    requests = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT_CARD_PATH = REPO_ROOT / "agent_card.toml"


def require_requests() -> None:
    if requests is None:
        print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
        sys.exit(1)


def run(cmd, **kwargs):
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def check_git_clean(force: bool) -> None:
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except SystemExit:
        print("ERROR: Must be run inside a git repository.", file=sys.stderr)
        sys.exit(1)
    if not force:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        if status.strip():
            print("ERROR: Working tree has uncommitted changes. Commit/stash or use --force.", file=sys.stderr)
            sys.exit(1)


def start_process(cmd) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_app(skip: bool) -> Optional[subprocess.Popen]:
    if skip:
        print("SKIP_APP_START=1 set; not starting Flask app.")
        return None
    print("Starting Flask assessor (python assessor/app.py)...")
    return start_process(["python", "assessor/app.py"])


def start_ngrok(port: str) -> subprocess.Popen:
    print(f"Starting ngrok on port {port}...")
    return start_process(["ngrok", "http", port])


def fetch_ngrok_url(timeout: int = 20) -> str:
    require_requests()
    api = "http://127.0.0.1:4040/api/tunnels"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(api, timeout=2)
            resp.raise_for_status()
        except Exception:
            time.sleep(1)
            continue
        data = resp.json()
        for tunnel in data.get("tunnels", []):
            url = tunnel.get("public_url", "")
            if url.startswith("https://"):
                print(f"Found ngrok public URL: {url}")
                return url
        time.sleep(1)
    print(f"ERROR: Timed out waiting for ngrok at {api}", file=sys.stderr)
    sys.exit(1)


def build_entrypoint(public_url: str) -> str:
    host = public_url.replace("https://", "", 1)
    return f"https://{host}/.well-known/agent-card.json"


def update_toml(path: pathlib.Path, entrypoint: str) -> None:
    if not path.exists():
        print(f"ERROR: Missing {path}", file=sys.stderr)
        sys.exit(1)
    text = path.read_text()
    line = f'entrypoint = "{entrypoint}"'
    entry_re = re.compile(r"^entrypoint\s*=.*$", re.MULTILINE)
    if entry_re.search(text):
        new_text = entry_re.sub(line, text, count=1)
    else:
        type_match = re.search(r"^type\s*=.*$", text, flags=re.MULTILINE)
        if type_match:
            pos = type_match.end()
            new_text = text[:pos] + "\n" + line + text[pos:]
        else:
            if not text.endswith("\n"):
                text += "\n"
            new_text = text + line + "\n"
    path.write_text(new_text)


def git_commit() -> None:
    run(["git", "add", str(AGENT_CARD_PATH)])
    run(["git", "commit", "-m", "CI: set agent_card entrypoint to ngrok URL"])


def kill_process(proc: Optional[subprocess.Popen]) -> None:
    if proc and proc.poll() is None:
        proc.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Ignore dirty working tree")
    parser.add_argument("--cleanup", action="store_true", help="Kill started processes on exit")
    parser.add_argument("--no-commit", action="store_true", help="Update file without git commit")
    args = parser.parse_args()

    check_git_clean(args.force)

    skip_app = os.environ.get("SKIP_APP_START") == "1"
    app_proc = start_app(skip_app)
    port = os.environ.get("NGROK_PORT", "8080")
    ngrok_proc = start_ngrok(port)
    public_url = fetch_ngrok_url()
    entrypoint = build_entrypoint(public_url)
    update_toml(AGENT_CARD_PATH, entrypoint)
    if not args.no_commit:
        git_commit()
    print(f"Updated entrypoint to: {entrypoint}")
    print(f"File: {AGENT_CARD_PATH}")

    if args.cleanup:
        kill_process(ngrok_proc)
        kill_process(app_proc)


if __name__ == "__main__":
    main()
