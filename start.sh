#!/usr/bin/env bash
# Start the API (port 8010) and the Streamlit UI (port 8501). Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "No .venv: see README 'Setup'"; exit 1; }
curl -sf -m 3 "${LLM_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null || echo "warning: Ollama isn't answering; start it first"
API_PORT=${API_PORT:-8010}
$PY -m uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" --log-level warning &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
for _ in $(seq 1 30); do curl -sf "http://localhost:$API_PORT/healthz" >/dev/null && break; sleep 0.5; done
echo "API on http://localhost:$API_PORT (docs: /docs)"
API_URL="http://localhost:$API_PORT" $PY -m streamlit run frontend/app.py --server.address localhost --server.headless true --browser.gatherUsageStats false
