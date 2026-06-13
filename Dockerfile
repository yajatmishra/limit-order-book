# ──────────────────────────────────────────────────────────────────────────────
#  Dockerfile — portable container for the Limit Order Book dashboard
#
#  Build & run locally:
#    docker build -t lob-dashboard .
#    docker run --rm -p 8050:8050 lob-dashboard
#    open http://localhost:8050
#
#  Render can also build from this Dockerfile (set runtime: docker), but the
#  default render.yaml uses the native Python runtime for faster builds.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8050

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Only the dashboard package and the WSGI shim are needed at runtime.
COPY python/ python/
COPY wsgi.py .

EXPOSE 8050

# Honour $PORT when present (Render/Fly/Cloud Run), default to 8050 locally.
CMD ["sh", "-c", "gunicorn wsgi:server --workers 1 --threads 8 --timeout 120 --preload --bind 0.0.0.0:${PORT:-8050}"]
