import os

# -------------------------
# Set cache folder (fix write permission issues)
# -------------------------
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache"

import torch
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model directory (only folder needed)
# -------------------------
model_dir = "mt_model"

# -------------------------
# Load tokenizer and model
# -------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# -------------------------
# FastAPI app setup
# -------------------------
app = FastAPI(title="English→Bangla Translator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request / Response Schemas
# -------------------------
class TranslateIn(BaseModel):
    text: str
    max_new_tokens: int = Field(128, ge=1, le=512)
    num_beams: int = Field(4, ge=1, le=10)

class TranslateOut(BaseModel):
    translation: str

class BatchTranslateIn(BaseModel):
    texts: List[str]
    max_new_tokens: int = 128
    num_beams: int = 4

class BatchTranslateOut(BaseModel):
    translations: List[str]

# -------------------------
# Translation logic
# -------------------------
def generate_translation(inputs: List[str], max_new_tokens: int, num_beams: int) -> List[str]:
    batch = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# -------------------------
# API endpoints
# -------------------------
@app.get("/")
def home():
    return {"message": "Welcome to English→Bangla Translator API!"}

@app.post("/translate", response_model=TranslateOut)
def translate(payload: TranslateIn):
    try:
        translation = generate_translation([payload.text], payload.max_new_tokens, payload.num_beams)[0]
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate_batch", response_model=BatchTranslateOut)
def translate_batch(payload: BatchTranslateIn):
    try:
        translations = generate_translation(payload.texts, payload.max_new_tokens, payload.num_beams)
        return {"translations": translations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Run app (for local testing)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

