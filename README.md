# AchievAI Server (Cursor Plugin)

## Overview

This repository provides a FastAPI-based HTTP server acting as a Cursor plugin for interacting with Deepseek AI models (7B and 33B) on an external Linux box.

## Features

- Supports Deepseek 7B in FP16 fully on GPU
- Supports Deepseek 33B in 4-bit with bitsandbytes, FP16 compute, and CPU offload
- Configurable via environment variables: `MODEL_NAME`, `MONGODB_URI`
- Large context support with truncation logging to MongoDB
- Uses RAM disk for HuggingFace cache and model offload

## Prerequisites

- Linux server (Ubuntu 24.04 LTS recommended)
- NVIDIA GPU with CUDA 12.8
- A RAM disk mounted (e.g., 64 GB at `/mnt/ramdisk`)
- Python 3.11, `venv`

## Setup

1. **Mount RAM disk** (if not already done):

   ```bash
   sudo mkdir -p /mnt/ramdisk
   sudo mount -t tmpfs -o size=64G tmpfs /mnt/ramdisk
   ```

2. **Clone the repository and run the build script**:

   ```bash
   git clone https://github.com/yourusername/cursor_plugin.git
   cd cursor_plugin
   ./build.sh
   ```

   Sample `build.sh` output:

   ```
   🚀 Starting build process for AchievAI Server (HTTP only)...
   🧠 Using RAM disk for HuggingFace cache: /mnt/ramdisk/huggingface
   📦 Offload folder set to: /mnt/ramdisk/offload
   ✅ MongoDB is running.
   🚀 Using model: deepseek-33b
   🧠 Hint: Model weights will now be cached in /mnt/ramdisk/huggingface
   💾 If you're using offloading, set your server.py config to use: /mnt/ramdisk/offload
   🎧 Launching HTTP Uvicorn on port 8000…
   ```

3. **Environment variables** (optional, defaults shown):

   ```bash
   export MODEL_NAME="deepseek-33b"            # or deepseek, gpt-3.5, etc.
   export MONGODB_URI="mongodb://localhost:27017"
   export HF_HOME="/mnt/ramdisk/huggingface"   # hint for Transformers cache
   ```

4. **Run the server**:

   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

- Forward your external IP’s port 8000 to the server to integrate with Cursor.
- API endpoints mirror OpenAI’s:
  - `POST /v1/chat/completions`
  - `POST /chat/completions`
  - `GET /v1/models`
  - `GET /v1/truncated/{conversation_id}`

## Sample Logs

```
🎧 Launching HTTP Uvicorn on port 8000…
INFO:     Will watch for changes in ['/home/richard/server']
INFO:     Started reloader process [13539] using StatReload
2025-04-28 14:03:11,409 INFO server: Loaded deepseek-33b with 4-bit quantization via bitsandbytes.
2025-04-28 14:03:11,409 INFO server: deepseek-33b context window = 16384 tokens
```

## License

MIT
