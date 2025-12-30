#!/usr/bin/env python3
"""Sandbox for safe execution of agent actions.

Provides path confinement and basic isolation for agent file operations
and command execution.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class Sandbox:
    """Isolated workspace for agent execution.

    Provides:
    - Temporary workspace directory
    - Path confinement (no .. escapes)
    - Offline environment variables (best effort)
    - Cleanup on exit
    """

    def __init__(self, scenario_name: str, scenario_targets_dir: Path):
        """Initialize sandbox.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario being run
        scenario_targets_dir : Path
            Path to scenario's targets/ directory to copy
        """
        self.scenario_name = scenario_name
        self.scenario_targets_dir = scenario_targets_dir
        self.workspace_dir: Optional[Path] = None
        self._created = False

    def __enter__(self):
        """Create sandbox workspace."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup sandbox workspace."""
        self.cleanup()

    def setup(self):
        """Create sandbox workspace and copy scenario files."""
        # Create temporary directory
        self.workspace_dir = Path(tempfile.mkdtemp(prefix=f"sandbox_{self.scenario_name}_"))
        self._created = True

        logger.info(f"Created sandbox workspace: {self.workspace_dir}")

        # Copy scenario targets/ directory
        if self.scenario_targets_dir.exists():
            target_dest = self.workspace_dir / "targets"
            shutil.copytree(self.scenario_targets_dir, target_dest)
            logger.info(f"Copied targets/ to sandbox")

        # Copy other scenario files if needed (manifest, etc.)
        scenario_dir = self.scenario_targets_dir.parent
        for file_name in ["manifest.json", "README.md", "baseline_test.py"]:
            src = scenario_dir / file_name
            if src.exists():
                shutil.copy2(src, self.workspace_dir / file_name)

    def cleanup(self):
        """Remove sandbox workspace."""
        if self._created and self.workspace_dir and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
            logger.info(f"Cleaned up sandbox: {self.workspace_dir}")
            self._created = False

    def validate_path(self, path: str | Path) -> Path:
        """Validate that path is within sandbox workspace.

        Parameters
        ----------
        path : str or Path
            Path to validate

        Returns
        -------
        Path
            Absolute path within workspace

        Raises
        ------
        ValueError
            If path escapes workspace
        """
        if self.workspace_dir is None:
            raise RuntimeError("Sandbox not set up. Call setup() first.")

        # Convert to Path
        p = Path(path)

        # If relative, make it relative to workspace
        if not p.is_absolute():
            p = self.workspace_dir / p

        # Resolve and check it's within workspace
        try:
            resolved = p.resolve()
            workspace_resolved = self.workspace_dir.resolve()

            if workspace_resolved not in resolved.parents and resolved != workspace_resolved:
                raise ValueError(f"Path {path} escapes sandbox workspace")

            return resolved
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path {path}: {e}")

    def read_file(self, path: str) -> str:
        """Read file from sandbox.

        Parameters
        ----------
        path : str
            Path to file (relative to workspace or absolute within workspace)

        Returns
        -------
        str
            File contents
        """
        resolved = self.validate_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not resolved.is_file():
            raise ValueError(f"Not a file: {path}")

        return resolved.read_text()

    def write_file(self, path: str, content: str):
        """Write file to sandbox.

        Parameters
        ----------
        path : str
            Path to file (relative to workspace or absolute within workspace)
        content : str
            Content to write
        """
        resolved = self.validate_path(path)

        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(content)
        logger.debug(f"Wrote file: {path}")

    def list_dir(self, path: str = ".") -> List[str]:
        """List directory contents.

        Parameters
        ----------
        path : str
            Directory path (relative to workspace)

        Returns
        -------
        list of str
            Directory contents
        """
        resolved = self.validate_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not resolved.is_dir():
            raise ValueError(f"Not a directory: {path}")

        return sorted([item.name for item in resolved.iterdir()])

    def run_command(self, cmd: List[str], timeout: int = 30) -> dict:
        """Run command in sandbox workspace.

        Parameters
        ----------
        cmd : list of str
            Command and arguments
        timeout : int
            Timeout in seconds (default 30)

        Returns
        -------
        dict
            Result with stdout, stderr, returncode, success
        """
        if self.workspace_dir is None:
            raise RuntimeError("Sandbox not set up. Call setup() first.")

        # Prepare environment with offline hints
        env = os.environ.copy()
        env.update({
            'NO_PROXY': '*',
            'no_proxy': '*',
            'PIP_NO_INDEX': '1',
            'CONDA_OFFLINE': '1',
        })

        logger.debug(f"Running command in sandbox: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'success': result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': f'Command timed out after {timeout}s',
                'returncode': -1,
                'success': False,
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': f'Command failed: {e}',
                'returncode': -1,
                'success': False,
            }

    def git_diff(self) -> str:
        """Get git diff if workspace is a git repo.

        Returns
        -------
        str
            Git diff output or empty string
        """
        result = self.run_command(['git', 'diff'], timeout=10)
        return result['stdout'] if result['success'] else ''

    def grep(self, pattern: str, path: str = ".") -> List[str]:
        """Search for pattern in files.

        Parameters
        ----------
        pattern : str
            Pattern to search for
        path : str
            Path to search in (file or directory)

        Returns
        -------
        list of str
            Matching lines with file:line format
        """
        resolved = self.validate_path(path)
        matches = []

        if resolved.is_file():
            try:
                content = resolved.read_text()
                for i, line in enumerate(content.split('\n'), 1):
                    if pattern in line:
                        matches.append(f"{resolved.name}:{i}: {line}")
            except (UnicodeDecodeError, PermissionError):
                pass
        elif resolved.is_dir():
            # Recursively search directory
            for file_path in resolved.rglob('*'):
                if file_path.is_file():
                    try:
                        content = file_path.read_text()
                        for i, line in enumerate(content.split('\n'), 1):
                            if pattern in line:
                                rel_path = file_path.relative_to(resolved)
                                matches.append(f"{rel_path}:{i}: {line}")
                    except (UnicodeDecodeError, PermissionError):
                        pass

        return matches


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test sandbox
    scenario_dir = Path("scenarios/supply_chain_poisoning")
    targets_dir = scenario_dir / "targets"

    if targets_dir.exists():
        with Sandbox("test", targets_dir) as sandbox:
            print(f"Workspace: {sandbox.workspace_dir}")
            print(f"Contents: {sandbox.list_dir()}")

            # Test file operations
            sandbox.write_file("test.py", "print('hello')")
            content = sandbox.read_file("test.py")
            print(f"File content: {content}")

            # Test command
            result = sandbox.run_command(["python", "test.py"])
            print(f"Command output: {result}")
