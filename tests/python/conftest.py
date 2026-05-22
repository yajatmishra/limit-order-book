"""
pytest configuration: add python/ to sys.path so tests can
import limit-order-book's Python packages without installation.
"""
import sys
import os

# Resolve <repo>/python/ and add it once
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
