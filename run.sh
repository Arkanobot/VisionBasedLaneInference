#!/usr/bin/env bash
# Launch the lane-inference web application.
#
#   ./run.sh              serve on this machine only
#   ./run.sh --lan        also reachable from a phone on the same wifi
#   ./run.sh --dev        Vite dev server with hot reload, API proxied
#
# Everything runs from inside this directory; nothing is installed globally.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST=127.0.0.1
MODE=serve

for a in "$@"; do
  case "$a" in
    --lan) HOST=0.0.0.0 ;;
    --dev) MODE=dev ;;
    --port=*) PORT="${a#*=}" ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

[ -d .venv ] || { echo "No .venv here. Create one and install lane_inference/requirements.txt"; exit 1; }

if [ "$MODE" = dev ]; then
  echo "API   → http://127.0.0.1:8000"
  echo "Front → http://localhost:5173  (hot reload)"
  .venv/bin/python -m uvicorn server.api:app --host 127.0.0.1 --port 8000 --reload &
  API=$!
  trap 'kill $API 2>/dev/null' EXIT
  cd web && npm run dev
  exit 0
fi

if [ ! -d web/dist ]; then
  echo "Building the frontend (first run only)…"
  (cd web && npm install --silent && npm run build)
fi

if [ "$HOST" = 0.0.0.0 ]; then
  IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)
  cat <<TXT

  ────────────────────────────────────────────────
   On this Mac : http://localhost:$PORT
   On a phone  : http://$IP:$PORT      (same wifi)

   Uploads and every tab work over plain HTTP.
   The camera button will not — browsers only allow
   camera access over HTTPS or on localhost.
  ────────────────────────────────────────────────

TXT
else
  echo
  echo "  http://localhost:$PORT"
  echo
fi

exec .venv/bin/python -m uvicorn server.api:app --host "$HOST" --port "$PORT"
