"""
WSGI entry point for the Limit Order Book dashboard.

Production servers (gunicorn, uWSGI) import the Flask ``server`` object from
here.  Render / Procfile / Docker all use::

    gunicorn wsgi:server

The dashboard package lives under ``python/``, so we put it on ``sys.path``
before importing.  Importing ``dashboard.app`` builds the synthetic session
and the static figures exactly once at process start (or once in the gunicorn
master when ``--preload`` is used, shared with workers via copy-on-write).
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from dashboard.app import server  # noqa: E402  (path set up above)

# Some WSGI servers look for ``application`` by convention.
application = server

__all__ = ["server", "application"]
