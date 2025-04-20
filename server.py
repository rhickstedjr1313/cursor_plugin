#!/usr/bin/env python3
# server.py

import os
import json
from typing import List, Dict, Generator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Model registry ─────────────────────────────────────────────────────────────────
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

# ─── Request schema for chat ──────────────────────────────────────────────────────────
class ChatReq(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 1.0
    max_tokens: int = 16_384
    stream: bool = False


def create_app():
    # Default model key
    model_key = os.getenv("MODEL_NAME", "deepseek-33b")
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    model_id = MODELS[model_key]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Try 4-bit quantization first
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
            offload_folder="offload",
            offload_state_dict=True,
            low_cpu_mem_usage=True
        )
        print("Loaded model with 4-bit quantization via bitsandbytes.")
    except Exception as e:
        # If quantization fails, fallback to FP16 with automatic CPU offloading
        print(f"Quantization failed ({e}), falling back to FP16 auto offload...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            low_cpu_mem_usage=True
        )

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        body = await request.body()
        print(f"➡️ {request.method} {request.url.path} — body={body!r}")
        response = await call_next(request)
        print(f"⬅️ {request.method} {request.url.path} — status={response.status_code}")
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

    @app.post("/v1/chat/completions")
    async def chat_v1(req: ChatReq):
        # Build prompt
        prompt = "".join(f"{m.get('role','user').capitalize()}: {m.get('content','')}\n" for m in req.messages) + "Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Trim to max context
        max_ctx = getattr(model.config, "max_position_embeddings", 16_384)
        if inputs["input_ids"].shape[1] > max_ctx:
            inputs["input_ids"] = inputs["input_ids"][..., -max_ctx:]
            inputs["attention_mask"] = inputs["attention_mask"][..., -max_ctx:]

        # Generate
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=(req.temperature>0),
                temperature=req.temperature or 1.0
            )

        # Decode
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Stream or normal response
        if req.stream:
            def event_stream():
                chunk = {"choices":[{"delta":{"content":text},"index":0,"finish_reason":"stop"}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {"id":"local-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

    @app.post("/chat/completions")
    async def chat_root(req: ChatReq):
        return await chat_v1(req)

    return app


# Instantiate app
app = create_app()

