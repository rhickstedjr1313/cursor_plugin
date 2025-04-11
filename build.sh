#!/bin/bash

echo "🚀 Starting build process for AchievAI Server on Linux..."

# Check for Python and pip
if ! command -v python3 &> /dev/null || ! command -v pip &> /dev/null; then
    echo "❌ ERROR: Python3 or pip is not installed."
    exit 1
fi

# Set up virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip and install PyTorch Nightly for CUDA 12.8 (for RTX 5090 support)
echo "📦 Installing PyTorch Nightly with CUDA 12.8 support..."
pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other required packages
REQUIRED_PACKAGES=("uvicorn" "fastapi" "transformers" "pymongo" "accelerate")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import ${package}" &> /dev/null; then
        echo "📦 Installing ${package}..."
        pip install ${package}
    fi
done

# Ensure server.py exists
if [ ! -f "server.py" ]; then
    echo "❌ ERROR: server.py not found!"
    exit 1
fi

# Check if mongod is available
if ! command -v mongod &> /dev/null; then
    echo "❌ ERROR: MongoDB is not installed or not in PATH."
    exit 1
fi

# Start MongoDB if not already running
if ! pgrep -x "mongod" > /dev/null; then
    echo "🚀 Starting MongoDB with systemd..."
    sudo systemctl start mongod

    sleep 2
    if ! pgrep -x "mongod" > /dev/null; then
        echo "❌ ERROR: MongoDB failed to start. Check with 'systemctl status mongod'"
        exit 1
    fi
fi

echo "✅ MongoDB is running."

# Use provided model name or default to 'deepseek'
MODEL_CHOICE=${1:-deepseek}
export MODEL_NAME="$MODEL_CHOICE"

echo "🚀 Running AchievAI Server with model: $MODEL_CHOICE on http://localhost:8000..."
uvicorn server:create_app --factory --host 0.0.0.0 --port 8000 --reload
