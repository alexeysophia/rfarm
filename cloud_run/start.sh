#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected GPU devices:"
  nvidia-smi --query-gpu=name --format=csv,noheader || nvidia-smi
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT}"
