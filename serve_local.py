import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Local Panacea Oversight Agent")

# Point this to the folder you extracted from the ZIP!
MODEL_PATH = "./content/panacea_oversight_model"

print("="*60)
print(f"Loading AI Model from {MODEL_PATH}...")
print("This may take a minute depending on your computer's specs.")
print("="*60)

try:
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("Downloading/Loading Base Model (Qwen2.5 1.5B)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", 
        torch_dtype=torch.float16,
        device_map="auto" # Automatically uses GPU if you have one, else CPU
    )
    
    print("Applying your trained RL Adapter weights...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    print("✅ Model loaded successfully!")
except ImportError:
    print("❌ Missing a required library. Run this command first:")
    print("pip install peft")
    exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nMake sure you extracted panacea_oversight_model.zip into the panacea folder!")
    exit(1)

SYSTEM_PROMPT = (
    "You are a hospital oversight AI agent. Your job is to review resource "
    "claims from specialist doctors and detect deception: ghost patients, "
    "inflation, masking, and collusion.\n"
    "Respond EXCLUSIVELY in this format:\n"
    "VERDICT: <APPROVED or REJECTED>\n"
    "REASONING: <your concise reasoning>"
)

class Req(BaseModel):
    prompt: str
    temperature: float = 0.1

@app.get("/health")
def health():
    return {"status": "ok", "model": "panacea-oversight-local"}

@app.post("/generate")
def generate(req: Req):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": req.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128, # Enough for verdict + reasoning
            temperature=req.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return {"text": response}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 LOCAL SERVER RUNNING!")
    print("The Panacea inference server is expecting to find this at:")
    print("http://localhost:8000/generate")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
