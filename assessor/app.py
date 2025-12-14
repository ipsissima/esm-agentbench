"""Expose the demo assessor Flask app for tests and CLI entrypoints.

The demo SWE test harness imports :mod:`assessor.app` to start a local
Flask server. The actual application instance lives in
``esmassessor.green_server``. Importing it here keeps the public import
stable without duplicating configuration.
"""

from esmassessor.green_server import app

__all__ = ["app"]
