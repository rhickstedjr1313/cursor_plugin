#!/usr/bin/env python3
# server.py

import os
import sys
import json
import logging
from typing import List, Dict
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient

# ─── Logging setup ───────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s"
))
logger.addHandler(handler)

# ─── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "deepseek":      "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "gpt-3.5":       "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "gpt-3.5-turbo": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "mistral":       "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3":          "microsoft/Phi-3-mini-128k-instruct",
    "gemma":         "google/gemma-7b-it",
    "mixtral":       "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama3-8b":     "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek-33b":  "deepseek-ai/deepseek-coder-33b-instruct"
}

# ─── Request schema for chat ────────────────────────────────────────────────────
class ChatReq(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 1.0
    max_tokens: int = 16_384
    stream: bool = False

def create_app():
    # Connect to MongoDB for logging truncated context
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri)
    db = client["chat_logs"]
    truncated_coll = db["truncated_context"]

    # Default model key
    model_key = os.getenv("MODEL_NAME", "deepseek-33b")
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    model_id = MODELS[model_key]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_key == "deepseek-33b":
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
                offload_folder="/mnt/ramdisk/offload",
                offload_state_dict=True,
                low_cpu_mem_usage=True
            )
            logger.info("Loaded deepseek-33b with 4-bit quantization via bitsandbytes.")
        except Exception as e:
            logger.warning(f"33B quantization failed ({e}), falling back to FP16 offload...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="/mnt/ramdisk/offload",
                offload_state_dict=True,
                low_cpu_mem_usage=True
            )
    else:
        logger.info(f"Loading {model_key} with FP16 precision (no quantization).")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="/mnt/ramdisk/offload",
            offload_state_dict=True,
            low_cpu_mem_usage=True
        )

    # Verify and log context window
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if not max_ctx or max_ctx < 1:
        raise RuntimeError("Invalid model.config.max_position_embeddings")
    logger.info(f"{model_key} context window = {max_ctx} tokens")

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── API-Key Verification ───────────────────────────────────────────────────────
    API_KEY = os.getenv("API_KEY", "LetMeIn")

    @app.middleware("http")
    async def verify_api_key(request: Request, call_next):
        # only protect chat endpoints
        if request.url.path.startswith("/v1/chat") or request.url.path.startswith("/chat"):
            auth = request.headers.get("Authorization", "")
            # support both “LetMeIn” and “Bearer LetMeIn”
            if auth.lower().startswith("bearer "):
                token = auth[7:]
            else:
                token = auth
            if token != API_KEY:
                return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        return await call_next(request)

    # ─── Request logging ────────────────────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        body = await request.body()
        logger.debug(f"➡️ {request.method} {request.url.path} — body={body!r}")
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Error during request: {e}")
            raise
        logger.debug(f"⬅️ {request.method} {request.url.path} — status={response.status_code}")
        return response

    @app.get("/")
    async def home():
        return {"message": f"{model_key} server up"}

    @app.get("/v1/models")
    async def list_models_v1():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local"} for m in MODELS]}

    @app.get("/v1/models/{model_id}")
    async def get_model_v1(model_id: str):
        if model_id not in MODELS:
            return JSONResponse(status_code=404, content={"error": "Not found"})
        return {"id": model_id, "object": "model", "owned_by": "local"}

    @app.get("/models")
    async def list_models_root():
        return await list_models_v1()

    @app.get("/models/{model_id}")
    async def get_model_root(model_id: str):
        return await get_model_v1(model_id)

    @app.get("/v1/truncated/{conversation_id}")
    async def get_truncated(conversation_id: str):
        docs = list(truncated_coll.find({"conversation_id": conversation_id}).sort("timestamp", 1))
        for d in docs:
            d["_id"] = str(d["_id"])
        return {"dropped_context": docs}

    @app.post("/v1/chat/completions")
    async def chat_v1(req: ChatReq):
        # Build prompt
        prompt = "".join(f"{m.get('role','user').capitalize()}: {m.get('content','')}\n"
                         for m in req.messages) + "Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Debug original length
        orig_len = inputs["input_ids"].shape[1]
        logger.debug(f"Original input token length: {orig_len}")

        # Reserve headroom
        reserved = 512
        input_limit = max_ctx - reserved
        logger.debug(f"Input limit (max_ctx {max_ctx} - reserved {reserved}) = {input_limit}")

        # Truncate if needed
        if orig_len > input_limit:
            to_trunc = orig_len - input_limit
            # Record truncated tokens
            full_ids = inputs["input_ids"][0].tolist()
            dropped_ids = full_ids[:to_trunc]
            dropped_text = tokenizer.decode(dropped_ids, skip_special_tokens=False)
            truncated_coll.insert_one({
                "conversation_id": req.messages[0].get("conversation_id"),
                "timestamp": datetime.utcnow(),
                "model": model_key,
                "dropped_token_ids": dropped_ids,
                "dropped_text": dropped_text
            })
            # Perform truncation
            inputs["input_ids"] = inputs["input_ids"][..., -input_limit:]
            inputs["attention_mask"] = inputs["attention_mask"][..., -input_limit:]
            logger.warning(f"Truncated {to_trunc} tokens; kept last {input_limit}")
            # Show last tokens snippet
            last_ids = inputs["input_ids"][0, -10:].tolist()
            snippet = tokenizer.decode(inputs["input_ids"][0, -10:], skip_special_tokens=False)
            logger.debug(f"Last 10 token IDs: {last_ids}")
            logger.debug(f"Last 10 tokens snippet: {snippet!r}")
            orig_len = input_limit

        # Ensure generation headroom
        available = max_ctx - orig_len
        if available <= 0:
            logger.error(f"No tokens available for generation: orig_len={orig_len}, max_ctx={max_ctx}")
            raise HTTPException(status_code=400, detail="Input too long to generate any tokens")
        max_gen = min(req.max_tokens, available)
        logger.debug(f"Generating up to {max_gen} new tokens (available headroom: {available})")

        # Generate
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                do_sample=(req.temperature > 0),
                temperature=req.temperature or 1.0
            )

        # Compute token usage with shape handling
        if isinstance(out, torch.Tensor) and out.ndim == 2:
            total_len = out.shape[1]
            seq = out[0]
        else:
            seq = out if isinstance(out, torch.Tensor) else torch.tensor(out)
            total_len = seq.shape[0]
        gen_tokens = total_len - orig_len
        logger.debug(f"Generated tokens: {gen_tokens}, total sequence length: {total_len}")

        # Decode
        text = tokenizer.decode(seq[orig_len:], skip_special_tokens=True)

        if req.stream:
            def event_stream():
                chunk = {"choices":[{"delta":{"content":text},"index":0,"finish_reason":"stop"}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {
            "id": "local-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": orig_len, "completion_tokens": gen_tokens, "total_tokens": total_len}
        }

    @app.post("/chat/completions")
    async def chat_root(req: ChatReq):
        return await chat_v1(req)

    return app

# Instantiate app
app = create_app()
