# Cursor Deepseek Plugin

A FastAPI-based “Cursor” plugin that serves Deepseek code-generation models (7B and 33B) on an external Linux box with GPU and high-RAM CPU offload.

## 🚀 Features

- **Dual‐mode model support**
  - **deepseek-7b**: full FP16 on GPU
  - **deepseek-33b**: 4-bit quantization via BitsAndBytes with FP16 compute and CPU offload (fallback to FP16 offload if needed)
- **High throughput**
  - Configured to use 32 CPU threads
  - Large `max_tokens` default (100 000) for long contexts
- **RAM‐disk offload**
  - Offloading intermediate tensors to a fast in-memory `tmpfs`
- **MongoDB logging**
  - Truncated context is logged for debugging

## 🛠️ Prerequisites

- **Linux** server with:
  - NVIDIA GPU (≥ 32 GB VRAM recommended)
  - ≥ 192 GB system RAM (for large contexts & offload)
- **Python 3.10+**
- **MongoDB** running locally or remotely
- **CUDA Toolkit 12.8** (for BitsAndBytes builds)

## 📦 Installation

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-org/cursor-deepseek-plugin.git
   cd cursor-deepseek-plugin
   ```

2. **Set up a Python virtual environment**
   ```bash
   ./build.sh [MODEL_NAME]
   ```
   - By default `MODEL_NAME=deepseek-33b`. You can override:
     ```bash
     ./build.sh deepseek
     ```

3. **(Optional) Create a RAM‐disk** for offload files
   ```bash
   sudo mkdir -p /mnt/ramdisk
   sudo mount -t tmpfs -o size=64G tmpfs /mnt/ramdisk
   ```
   Adjust `size=` up to your available RAM (e.g. `size=96G`).

4. **Configure environment variables**
   ```bash
   export MONGODB_URI="mongodb://localhost:27017"
   export MODEL_NAME="deepseek-33b"
   ```

## ⚙️ Configuration

All options are in `server.py` or via env vars:

| Setting                    | Env var       | Default                |
|----------------------------|---------------|------------------------|
| Model to load              | `MODEL_NAME`  | `deepseek-33b`         |
| MongoDB URI                | `MONGODB_URI` | `mongodb://localhost:27017` |
| Offload folder             | —             | `/mnt/ramdisk/offload` |
| CPU threads                | —             | `torch.set_num_threads(32)` |
| Default max tokens per req | —             | `100_000`              |

## 🚨 Running the Server

```bash
uvicorn server:app   --host 0.0.0.0 --port 8000 --reload
```

On startup you’ll see logs like:

```
Loaded deepseek-33b with 4-bit quantization via bitsandbytes.
deepseek-33b context window = 32768 tokens
```

## 📝 API Reference

### GET /v1/models

List available models.

### GET /v1/models/{model_id}

Details for a specific model.

### POST /v1/chat/completions

Request schema:
```json
{
  "model": "deepseek-33b",
  "messages": [
    { "role": "user", "content": "Hello, world!" }
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

Response:
```json
{
  "id":"local-1",
  "object":"chat.completion",
  "choices":[
    {
      "index":0,
      "message":{"role":"assistant","content":"Hi there!"},
      "finish_reason":"stop"
    }
  ],
  "usage":{
    "prompt_tokens":5,
    "completion_tokens":4,
    "total_tokens":9
  }
}
```

If you set `"stream": true`, the server will send an SSE stream of partial tokens.

## 🙏 Contributing

1. Fork & clone  
2. Create a feature branch  
3. Submit a PR against `main`  
4. Ensure linting, formatting, and tests pass

## 📄 License

Released under the MIT License.
