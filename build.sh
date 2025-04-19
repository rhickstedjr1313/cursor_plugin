#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

echo "🚀 Starting build process for AchievAI Server (HTTP only)..."

# 1) venv & activate
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment…"
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# 2) Dependencies
echo "📦 Installing/upgrading pip & core libraries…"
pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
for pkg in uvicorn fastapi transformers pymongo accelerate; do
  pip install --upgrade "$pkg"
done

# 3) Check server.py
if ! grep -qE '^app\s*=\s*create_app\(\)' server.py; then
  echo "❌ ERROR: server.py must include at top‑level: app = create_app()"
  exit 1
fi

# 4) Ensure MongoDB is running
if ! pgrep -x mongod &>/dev/null; then
  echo "🚀 Starting MongoDB…"
  sudo systemctl start mongod
  sleep 2
  if ! pgrep -x mongod &>/dev/null; then
    echo "❌ ERROR: Unable to start mongod. Check 'systemctl status mongod'."
    exit 1
  fi
fi
echo "✅ MongoDB is running."

# 5) Model selection
MODEL_NAME="${1:-deepseek}"
export MODEL_NAME
echo "🚀 Using model: $MODEL_NAME"

# 6) Launch HTTP only
echo "🎧 Launching HTTP Uvicorn on port 8000…"
uvicorn server:app \
  --host 0.0.0.0 --port 8000 --reload
