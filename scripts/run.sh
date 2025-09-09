#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [ ! -f .env ]; then
  echo ".env not found. Create it with API keys and settings."
fi

exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080


