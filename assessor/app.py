"""Compatibility module for importing the Flask app.

The Flask app is defined in esmassessor.green_server, but for backwards
compatibility with existing tests, we re-export it here.
"""
from esmassessor.green_server import app

__all__ = ["app"]
