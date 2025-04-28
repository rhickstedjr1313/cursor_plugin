#!/usr/bin/env bash
#
export OPENAI_API_KEY=LetMeIn

set -euo pipefail
IFS=$'\n\t'

echo "🚀 Starting build process for AchievAI Server (HTTP only)..."

# 0) RAM disk offload + cache setup
RAMDISK_DIR="/mnt/ramdisk"
HF_CACHE="$RAMDISK_DIR/huggingface"
OFFLOAD_DIR="$RAMDISK_DIR/offload"
mkdir -p "$HF_CACHE" "$OFFLOAD_DIR"
echo "🧠 Using RAM disk for HuggingFace cache: $HF_CACHE"
echo "📦 Offload folder set to: $OFFLOAD_DIR"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE"

# 1) venv & activate
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment…"
  python3 -m venv venv
fi
source venv/bin/activate

# 2) Dependencies
echo "📦 Installing/upgrading pip & core libraries…"
pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
for pkg in uvicorn fastapi transformers pymongo accelerate bitsandbytes; do
  pip install --upgrade "$pkg"
done

# 2a) Build bitsandbytes for CUDA 12.8 if not already installed
if ! python -c "import bitsandbytes" &>/dev/null; then
  echo "🔧 Cloning bitsandbytes for CUDA build…"
  git clone https://github.com/TimDettmers/bitsandbytes.git tmp_bnb
  cd tmp_bnb
  echo "🔨 Building bitsandbytes (CUDA_VERSION=128)…"
  CUDA_VERSION=128 python setup.py install
  cd ..
  rm -rf tmp_bnb
else
  echo "✅ bitsandbytes already installed in venv"
fi

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
MODEL_NAME="${1:-deepseek-33b}"
export MODEL_NAME
echo "🚀 Using model: $MODEL_NAME"

# 6) RAM disk offload + model caching hint
echo "🧠 Hint: Model weights will now be cached in $HF_CACHE"
echo "💾 If you're using offloading, set your server.py config to use: $OFFLOAD_DIR"

# 7) Launch HTTP only
echo "🎧 Launching HTTP Uvicorn on port 8000…"
uvicorn server:app \
  --host 0.0.0.0 --port 8000 --reload
