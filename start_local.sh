#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
export GRADIO_ANALYTICS_ENABLED=False
python check_deployment.py
exec python -u app.py
