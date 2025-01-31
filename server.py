import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlx.core as mx
import torch

# Set MLX to use GPU
mx.set_default_device(mx.gpu)

# Model name from Hugging Face
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"

# Define an offload folder to handle disk memory usage
OFFLOAD_FOLDER = "./offload_weights"
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with FP16 precision instead of bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # ✅ Use FP16 (works on Apple Silicon)
    device_map="auto",
    offload_folder=OFFLOAD_FOLDER  # Store offloaded weights here
)

print("Model loaded successfully!")

# Initialize FastAPI
app = FastAPI()

# Define request schema
class RequestBody(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate_text(request: RequestBody):
    """Generates text using DeepSeek AI and logs request & inference time."""
    
    total_start_time = time.time()  # Start timing full request

    inputs = tokenizer(request.prompt, return_tensors="pt").to("mps")  # Apple GPU (MPS backend)

    model_start_time = time.time()  # Start timing model inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=0.7,  # Adjust as needed
            top_p=0.9
        )
    model_end_time = time.time()  # End timing model inference

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    total_end_time = time.time()  # End timing full request

    model_time = model_end_time - model_start_time
    total_time = total_end_time - total_start_time

    print(f"🕒 Model Inference Time: {model_time:.4f} seconds")
    print(f"⏳ Total Request Time: {total_time:.4f} seconds")

    return {
        "response": response_text,
        "timing": {
            "model_inference_time": model_time,
            "total_request_time": total_time
        }
    }

@app.get("/")
def home():
    return {"message": "DeepSeek AI Model Server is Running on macOS!"}
