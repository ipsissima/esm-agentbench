"""Tests for kernel dependency checking with macOS/Windows support.

This test suite validates that kernel dependency checks work across platforms.
"""
from __future__ import annotations

import sys
import pytest
from unittest.mock import patch, MagicMock
import subprocess

from certificates.verified_kernel import (
    _check_library_dependencies,
    _parse_ldd_output,
    _parse_otool_output,
)

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_parse_ldd_output_with_missing_deps():
    """Test parsing ldd output with missing dependencies."""
    output = """
        linux-vdso.so.1 (0x00007fff123456)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f1234567890)
        libfoo.so => not found
        libbar.so.1 => not found
        /lib64/ld-linux-x86-64.so.2 (0x00007f9876543210)
    """
    
    all_ok, missing = _parse_ldd_output(output)
    
    assert not all_ok
    assert len(missing) == 2
    assert any("libfoo.so" in dep for dep in missing)
    assert any("libbar.so.1" in dep for dep in missing)


def test_parse_ldd_output_all_satisfied():
    """Test parsing ldd output with all dependencies satisfied."""
    output = """
        linux-vdso.so.1 (0x00007fff123456)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f1234567890)
        /lib64/ld-linux-x86-64.so.2 (0x00007f9876543210)
    """
    
    all_ok, missing = _parse_ldd_output(output)
    
    assert all_ok
    assert len(missing) == 0


def test_parse_otool_output_normal():
    """Test parsing otool -L output with normal dependencies."""
    output = """
/usr/local/lib/libfoo.dylib:
    /usr/local/lib/libfoo.dylib (compatibility version 1.0.0, current version 1.2.3)
    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1281.100.1)
    """
    
    all_ok, missing = _parse_otool_output(output)
    
    assert all_ok
    assert len(missing) == 0


def test_parse_otool_output_with_error():
    """Test parsing otool -L output with error messages."""
    output = """
/usr/local/lib/libfoo.dylib:
    error: library not found
    /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1281.100.1)
    """
    
    all_ok, missing = _parse_otool_output(output)
    
    assert not all_ok
    assert len(missing) == 1


@patch('sys.platform', 'linux')
@patch('subprocess.run')
def test_check_library_dependencies_linux(mock_run):
    """Test that Linux uses ldd."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x123)\n"
    )
    
    all_ok, msg = _check_library_dependencies("/path/to/kernel.so")
    
    # Should call ldd
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "ldd"
    assert args[1] == "/path/to/kernel.so"
    
    assert all_ok
    assert "All dependencies satisfied" in msg


@patch('sys.platform', 'darwin')
@patch('subprocess.run')
def test_check_library_dependencies_macos(mock_run):
    """Test that macOS uses otool -L."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n"
    )
    
    all_ok, msg = _check_library_dependencies("/path/to/kernel.dylib")
    
    # Should call otool
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "otool"
    assert args[1] == "-L"
    assert args[2] == "/path/to/kernel.dylib"
    
    assert all_ok
    assert "All dependencies satisfied" in msg


@patch('sys.platform', 'win32')
def test_check_library_dependencies_windows():
    """Test that Windows returns a skip message."""
    all_ok, msg = _check_library_dependencies("C:\\path\\to\\kernel.dll")
    
    # Windows should skip with a message
    assert all_ok
    assert "Windows" in msg
    assert "manual verification required" in msg


@patch('sys.platform', 'unknown-os')
def test_check_library_dependencies_unknown_platform():
    """Test that unknown platforms return a skip message."""
    all_ok, msg = _check_library_dependencies("/path/to/kernel.so")
    
    # Unknown platform should skip with a message
    assert all_ok
    assert "unknown platform" in msg.lower()


@patch('sys.platform', 'linux')
@patch('subprocess.run')
def test_check_library_dependencies_tool_not_found(mock_run):
    """Test handling when dependency checking tool is not available."""
    mock_run.side_effect = FileNotFoundError("ldd not found")
    
    all_ok, msg = _check_library_dependencies("/path/to/kernel.so")
    
    # Should return success (skip) when tool not available
    assert all_ok
    assert "check skipped" in msg.lower()


@patch('sys.platform', 'linux')
@patch('subprocess.run')
def test_check_library_dependencies_with_missing_deps(mock_run):
    """Test detection of missing dependencies on Linux."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="libfoo.so => not found\nlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6\n"
    )
    
    all_ok, msg = _check_library_dependencies("/path/to/kernel.so")
    
    assert not all_ok
    assert "Missing dependencies" in msg
    assert "libfoo.so" in msg
