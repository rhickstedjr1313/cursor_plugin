#!/bin/bash

echo "🚀 Starting build process for AchievAI Server..."

# Ensure Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ ERROR: Python3 is not installed! Please install it before proceeding."
    exit 1
fi

# Ensure pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "❌ ERROR: pip3 is not installed! Please install it before proceeding."
    exit 1
fi

# Ensure Uvicorn is installed
if ! python3 -c "import uvicorn" &> /dev/null
then
    echo "📦 Installing Uvicorn..."
    pip3 install uvicorn fastapi transformers torch
fi

# Ensure the server file exists
if [ ! -f "server.py" ]; then
    echo "❌ ERROR: server.py not found! Make sure you are in the correct directory."
    exit 1
fi

# Run the FastAPI server
echo "🚀 Running AchievAI Server on http://localhost:8000..."
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
