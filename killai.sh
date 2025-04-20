#!/usr/bin/env bash
set -euo pipefail

# Default port is 8000; override with first argument:
PORT="${1:-8000}"

# Find PID(s) listening on that port
PIDS=$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN -Pn || true)

if [[ -z "$PIDS" ]]; then
  echo "✅ No process is listening on port $PORT"
  exit 0
fi

echo "⚠️  Found process(es) on port $PORT: $PIDS"
for PID in $PIDS; do
  echo "🛑 Sending SIGTERM to PID $PID..."
  kill "$PID" || true

  # Give it a moment to exit
  sleep 2

  # If still running, force‑kill
  if kill -0 "$PID" &>/dev/null; then
    echo "💥 PID $PID did not exit—sending SIGKILL"
    kill -9 "$PID" || true
  else
    echo "✅ PID $PID terminated"
  fi
done

echo "🎉 Port $PORT is now free."
