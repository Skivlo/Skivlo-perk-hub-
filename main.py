import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load environment variables securely from .env file
load_dotenv()

# 2. Configure Gemini AI using the secure environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 3. Initialize FastAPI App
app = FastAPI(
    title="Skivlo Perk Hub Backend",
    description="Production-grade AI Backend Engine",
    version="1.0.0"
)

# Request schema for AI Chat
class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Engine is running securely!"}

@app.post("/api/ai/chat")
async def generate_ai_response(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key is not configured.")
    try:
        # Standard production implementation for Gemini Pro
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(request.prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
