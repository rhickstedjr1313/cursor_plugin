# server.py

import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pymongo import MongoClient

MODELS = {
    "deepseek": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",  # ✅ Updated model
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3": "microsoft/Phi-3-mini-128k-instruct",
    "gemma": "google/gemma-7b-it",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct"
}

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client.deepseek_db
conversations_collection = db.conversations

def save_conversation(user_id, conversation):
    conversations_collection.update_one(
        {"user_id": user_id},
        {"$set": {"conversation": conversation}},
        upsert=True
    )

def load_conversation(user_id):
    user_data = conversations_collection.find_one({"user_id": user_id})
    return user_data["conversation"] if user_data else []

def create_app():
    model_name = os.environ.get("MODEL_NAME", "deepseek")
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {', '.join(MODELS.keys())}")
    MODEL_ID = MODELS[model_name]

    print(f"🚀 Loading model '{model_name}' from '{MODEL_ID}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": 0}  # ✅ Force full model to GPU 0
    )
    print("✅ Model loaded successfully!")

    app = FastAPI()

    class RequestBody(BaseModel):
        user_id: str
        message: str
        max_tokens: int = 100

    @app.post("/generate")
    async def generate_text(request: Request):
        try:
            body = await request.json()
            print("✅ Received JSON:", body)
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            return JSONResponse(status_code=400, content={"error": "Invalid JSON format", "details": str(e)})

        try:
            req = RequestBody(**body)
        except Exception as e:
            print("❌ Schema validation failed:", e)
            return JSONResponse(status_code=422, content={"error": "Validation error", "details": str(e)})

        user_id = req.user_id
        conversation_history = load_conversation(user_id)
        conversation_history.append(f"User: {req.message}")
        conversation_history = conversation_history[-10:]
        context = "\n".join(conversation_history) + "\nAssistant:"
        print(f"🧠 Full context for user '{user_id}':\n{context}\n")

        total_start = time.time()
        inputs = tokenizer(context, return_tensors="pt").to(model.device)

        if inputs["input_ids"].shape[1] > 2048:
            inputs["input_ids"] = inputs["input_ids"][:, -2048:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -2048:]
            print("⚠️ Input truncated to 2048 tokens.")

        model_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=0.7,
                top_p=0.9
            )
        model_end = time.time()

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_end = time.time()

        print(f"✅ Response generated in {model_end - model_start:.2f}s (Total: {total_end - total_start:.2f}s)")
        print(f"🗣️ Assistant:\n{response_text.strip()}\n")

        conversation_history.append(f"Assistant: {response_text}")
        save_conversation(user_id, conversation_history)

        return {
            "response": response_text,
            "timing": {
                "model_inference_time": model_end - model_start,
                "total_request_time": total_end - total_start
            }
        }

    @app.get("/")
    def home():
        return {"message": f"{model_name} Model Server is Running with CUDA and MongoDB!"}

    return app

# Expose app for uvicorn
app = create_app()
