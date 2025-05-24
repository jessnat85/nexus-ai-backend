import os
import io
import base64
import json
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    strategy: str
    signal: str
    bias: str
    pattern: str
    entry: float
    stopLoss: float
    takeProfit: float
    confidence: float

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_chart(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a financial trading expert AI. Analyze this trading chart image, "
                                "and return the strategy, signal, bias, pattern, entry price, stop loss (SL), take profit (TP), "
                                "and confidence percentage in JSON format.\n"
                                "Format your reply ONLY as JSON like this:\n"
                                "{\"strategy\": \"SMC\", \"signal\": \"BUY\", \"bias\": \"Bullish\", \"pattern\": \"Order Block\", "
                                "\"entry\": 2315.55, \"stopLoss\": 2301.25, \"takeProfit\": 2348.95, \"confidence\": 92.5}\n"
                                "Do NOT explain anything else."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        raw = response.choices[0].message.content
        print("GPT raw output:", raw)
        json_str = re.search(r'{.*}', raw, re.DOTALL)
        if not json_str:
            raise ValueError("No JSON found in response")

        parsed = json.loads(json_str.group())
        return AnalysisResult(**parsed)

    except Exception as e:
        print("Error during GPT Vision analysis:", str(e))
        return AnalysisResult(strategy="N/A", signal="", bias="", pattern="", entry=0, stopLoss=0, takeProfit=0, confidence=0)
