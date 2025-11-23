#!/usr/bin/env python3
"""Extract the local AgentBeats tutorial archive into third_party/tutorial."""
from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)
    LOGGER.info("extracted %s to %s", archive_path, target_dir)


def ensure_layout(target_dir: Path) -> None:
    agentbeats_dir = target_dir / "src" / "agentbeats"
    debate_dir = target_dir / "scenarios" / "debate"
    if not agentbeats_dir.exists() or not debate_dir.exists():
        raise RuntimeError("tutorial archive missing expected src/agentbeats or scenarios/debate")
    (target_dir / "agentbeats").mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import AgentBeats tutorial archive")
    parser.add_argument("--archive", type=Path, default=Path("/mnt/data/tutorial-main.zip"))
    parser.add_argument("--target", type=Path, default=Path("third_party/tutorial"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not args.archive.exists():
        LOGGER.warning("Archive not found at %s; assuming tutorial already present", args.archive)
        ensure_layout(args.target)
        return
    extract_archive(args.archive, args.target)
    ensure_layout(args.target)


if __name__ == "__main__":
    main()
