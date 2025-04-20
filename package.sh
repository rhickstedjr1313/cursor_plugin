#!/usr/bin/env bash

echo "🔄 Generating requirements.txt..."
pip freeze > requirements.txt

echo "🧹 Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "📦 Creating server_package.tgz..."
tar -czf server_package.tgz \
  server.py \
  build.sh \
  requirements.txt \
  # add other files/folders you need (e.g., .env, offload_weights, etc.)

echo "✅ Package created: server_package.tgz"
