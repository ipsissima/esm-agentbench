#!/usr/bin/env python3
"""Tool definitions for local coding agents.

Provides a minimal but sufficient set of tools for coding tasks:
- read_file: Read file contents
- write_file: Write file contents
- list_dir: List directory contents
- run_cmd: Execute shell commands (sandboxed)
- git_diff: Show git diff
- grep: Search for patterns in files

All tools operate within the sandbox workspace.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List

from .sandbox import Sandbox

logger = logging.getLogger(__name__)


class Tool:
    """Base class for agent tools."""

    def __init__(self, name: str, description: str, parameters: Dict[str, str]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, sandbox: Sandbox, args: Dict[str, Any]) -> str:
        """Execute the tool.

        Parameters
        ----------
        sandbox : Sandbox
            Sandbox to execute in
        args : dict
            Tool arguments

        Returns
        -------
        str
            Tool execution result
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for prompt generation."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }


class ReadFileTool(Tool):
    """Tool for reading file contents."""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "path": "Path to the file to read (relative to workspace)",
            }
        )

    def execute(self, sandbox: Sandbox, args: Dict[str, Any]) -> str:
        try:
            path = args['path']
            content = sandbox.read_file(path)
            return f"File: {path}\n\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    """Tool for writing file contents."""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file",
            parameters={
                "path": "Path to the file to write (relative to workspace)",
                "content": "Content to write to the file",
            }
        )

    def execute(self, sandbox: Sandbox, args: Dict[str, Any]) -> str:
        try:
            path = args['path']
            content = args['content']
            sandbox.write_file(path, content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class ListDirTool(Tool):
    """Tool for listing directory contents."""

    def __init__(self):
        super().__init__(
            name="list_dir",
            description="List contents of a directory",
            parameters={
                "path": "Path to the directory (default: current directory)",
            }
        )

    def execute(self, sandbox: Sandbox, args: Dict[str, Any]) -> str:
        try:
            path = args.get('path', '.')
            contents = sandbox.list_dir(path)
            return f"Directory: {path}\n\n" + "\n".join(contents)
        except Exception as e:
            return f"Error listing directory: {e}"


class RunCmdTool(Tool):
    """Tool for running shell commands."""

    def __init__(self):
        super().__init__(
            name="run_cmd",
            description="Run a shell command in the workspace",
            parameters={
                "cmd": "Command as list of strings (e.g., ['pytest', '-q'])",
            }
        )

    def execute(self, sandbox: Sandbox, args: Dict[str, Any]) -> str:
        try:
            cmd = args['cmd']
            if isinstance(cmd, str):
                # If string provided, split it
                cmd = cmd.split()

            result = sandbox.run_command(cmd)

            output = []
            if result['stdout']:
                output.append(f"STDOUT:\n{result['stdout']}")
            if result['stderr']:
                output.append(f"STDERR:\n{result['stderr']}")
            output.append(f"Return code: {result['returncode']}")

            return "\n\n".join(output)
        except Exception as e:
            return f"Error running command: {e}"


class GitDiffTool(Tool):
    """Tool for showing git diff."""

    def __init__(self):
        super().__init__(
            name="git_diff",
            description="Show git diff of changes in the workspace",
            parameters={}
        )

    def execute(self, sandbox: Sandbox, args: Dict[str, Any]) -> str:
        try:
            diff = sandbox.git_diff()
            if diff:
                return f"Git diff:\n\n{diff}"
            else:
                return "No git diff available (not a git repo or no changes)"
        except Exception as e:
            return f"Error getting git diff: {e}"


class GrepTool(Tool):
    """Tool for searching files."""

    def __init__(self):
        super().__init__(
            name="grep",
            description="Search for a pattern in files",
            parameters={
                "pattern": "Pattern to search for",
                "path": "Path to search in (file or directory, default: current directory)",
            }
        )

    def execute(self, sandbox: Sandbox, args: Dict[str, Any]) -> str:
        try:
            pattern = args['pattern']
            path = args.get('path', '.')
            matches = sandbox.grep(pattern, path)

            if matches:
                return f"Found {len(matches)} matches:\n\n" + "\n".join(matches[:50])
            else:
                return f"No matches found for pattern: {pattern}"
        except Exception as e:
            return f"Error searching: {e}"


# Tool registry
TOOLS = [
    ReadFileTool(),
    WriteFileTool(),
    ListDirTool(),
    RunCmdTool(),
    GitDiffTool(),
    GrepTool(),
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}


def get_tool(name: str) -> Tool:
    """Get tool by name.

    Parameters
    ----------
    name : str
        Tool name

    Returns
    -------
    Tool
        Tool instance

    Raises
    ------
    ValueError
        If tool not found
    """
    if name not in TOOL_MAP:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOL_MAP.keys())}")
    return TOOL_MAP[name]


def execute_tool(sandbox: Sandbox, tool_name: str, args: Dict[str, Any]) -> str:
    """Execute a tool by name.

    Parameters
    ----------
    sandbox : Sandbox
        Sandbox to execute in
    tool_name : str
        Name of the tool
    args : dict
        Tool arguments

    Returns
    -------
    str
        Tool execution result
    """
    tool = get_tool(tool_name)
    return tool.execute(sandbox, args)


def generate_tool_docs() -> str:
    """Generate documentation for all tools.

    Returns
    -------
    str
        Tool documentation for prompt
    """
    docs = ["Available tools:"]
    for tool in TOOLS:
        docs.append(f"\n{tool.name}:")
        docs.append(f"  Description: {tool.description}")
        if tool.parameters:
            docs.append("  Parameters:")
            for param, desc in tool.parameters.items():
                docs.append(f"    - {param}: {desc}")
    return "\n".join(docs)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(generate_tool_docs())
