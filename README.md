# Cursor Deepseek Plugin

A Cursor plugin for integrating Deepseek 7B/33B models on an external Linux server.

## Overview

This plugin allows Cursor to send chat/completion requests to a FastAPI server running Deepseek models (7B in FP16 or 33B in 4‑bit + FP16 compute with CPU offload) hosted on an external Linux box.

## Supported Models

- **Deepseek 7B (FP16)**  
  Hugging Face ID: `deepseek-ai/deepseek-coder-7b-instruct-v1.5`
- **Deepseek 33B (4‑bit + FP16 compute + CPU offload)**  
  Hugging Face ID: `deepseek-ai/deepseek-coder-33b-instruct`

## Requirements

- Linux server with:
  - NVIDIA GPU (32 GB or larger)
  - At least 64 GB RAM (configurable RAM disk)
  - CUDA 12.8
- Python 3.11+
- Git
- `venv` or `virtualenv`
- Port 8000 accessible externally

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/cursor-deepseek-plugin.git
   cd cursor-deepseek-plugin
   ```

2. **Setup virtual environment & install dependencies**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Build bitsandbytes (if needed)**  
   ```bash
   ./build.sh
   ```

## Configuration

1. **Environment Variables**  
   - `MODEL_NAME`: Choose between `deepseek` (7B) or `deepseek-33b`  
   - `MONGODB_URI`: MongoDB connection string for logging (default: `mongodb://localhost:27017`)

2. **RAM Disk for Offloading**  
   By default, offload state dicts to `/mnt/ramdisk/offload`.  
   Ensure you have a RAM disk mounted:
   ```bash
   sudo mount -t tmpfs -o size=64G tmpfs /mnt/ramdisk
   ```

3. **Expose Server Externally**  
   - **Find External IP**:  
     ```bash
     curl ifconfig.me
     ```
   - **Port Forwarding**:  
     On your router, forward external port **8000** to the server’s local IP on port **8000**.

## Running the Server

```bash
source venv/bin/activate
MODEL_NAME=deepseek-33b uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Cursor Integration

In your Cursor settings, point the AI endpoint to:

```
http://<YOUR_EXTERNAL_IP>:8000/v1/chat/completions
```

## Usage

Use your Cursor interface to interact with the Deepseek models as usual. Ensure `MODEL_NAME` is set appropriately per session.

## License

MIT License
