# Deepseek Cursor Plugin

A Cursor extension for running Deepseek 7B/33B models on an external Linux server.

## Features

- Hosts Deepseek-7B and Deepseek-33B on your own server
- FP16 on GPU for 7B
- 4‑bit quant + FP16 compute with CPU offload for 33B
- Customizable max tokens, CPU threading, and RAM‑disk offloading
- Simple OpenAI‑compatible HTTP API

## Installation

```bash
git clone https://github.com/yourusername/deepseek-cursor-plugin.git
cd deepseek-cursor-plugin
./build.sh deepseek-33b
```

## Configuration

1. **Set Your API Key**  
   The server expects an `Authorization` header matching the `API_KEY` environment variable.  
   By default:
   ```bash
   export API_KEY=LetMeIn
   ```
2. **Override OpenAI Base URL**  
   In Cursor, go to **Settings → OpenAI API Key**, toggle on, enter your key, then under _Override OpenAI Base URL_ set:
   ```
   http://<your_ext_ip>:8000
   ```
   ![OpenAI API Key Config](Screenshot 2025-04-28 at 1.17.19 PM.jpeg)

3. **Server Settings**  
   - Server listens on port `8000`  
   - Ensure port forwarding from your router to the server's local IP  
   - Optionally adjust `offload_folder` to your RAM‑disk path (e.g. `/mnt/ramdisk/offload`)

## Usage

Start the server:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Use it in Cursor by selecting your custom model (e.g. `deepseek-33b`) in the model list.

## License

MIT
