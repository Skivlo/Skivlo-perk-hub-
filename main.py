import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

load_dotenv()

# API Keys Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Skivlo Perk Hub - AI Tools Discovery Engine")

# User kya tool dhundhna chahta hai, uski request
class ToolSearchRequest(BaseModel):
    user_requirement: str  # Misaal ke taur par: "Video editing tool for free"
    search_type: str = "quick"  # "quick" matlb flash/mini, "deep" matlb pro/gpt4

@app.post("/api/tools/find")
async def find_ai_tools(request: ToolSearchRequest):
    req = request.user_requirement
    
    # System Prompt: AI ko batana ki uska kaam AI Tools dhundhna hai
    system_instruction = (
        "You are the core engine of Skivlo Perk Hub. Your job is to find, recommend, "
        "and compare the best AI tools based on the user's requirements. Provide precise, "
        "unbiased recommendations, pricing details, and alternatives."
    )

    # --- 1. QUICK SEARCH (Gemini Flash ya GPT Mini) ---
    if request.search_type == "quick":
        # Agar OpenAI key hai toh GPT Mini chalao, nahi toh Gemini Flash
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"Find best tools for: {req}"}
                    ]
                )
                return {"tools": response.choices[0].message.content, "engine": "GPT-4o-Mini"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=500, detail="Gemini Key missing!")
            try:
                model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)
                response = model.generate_content(req)
                return {"tools": response.text, "engine": "Gemini-1.5-Flash"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    # --- 2. DEEP COMPARISON & EXPERT FINDER (Gemini Pro ya GPT-4) ---
    elif request.search_type == "deep":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini Pro Key is required for Deep analysis!")
        try:
            # Deep research ke liye Gemini Pro ka use
            model = genai.GenerativeModel("gemini-1.5-pro", system_instruction=system_instruction)
            response = model.generate_content(f"Do a deep expert analysis and find the ultimate AI tools for: {req}")
            return {"tools": response.text, "engine": "Gemini-1.5-Pro"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(status_code=400, detail="Invalid search type! Choose 'quick' or 'deep'.")
