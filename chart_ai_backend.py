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
    commentary: str

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
                                "You are an expert SMC (Smart Money Concepts) trading assistant. Analyze the chart and identify the following ONLY if visible:\n"
                                "- CHoCH (Change of Character) or BOS (Break of Structure)\n"
                                "- Supply or Demand Order Blocks (OB)\n"
                                "- Fair Value Gaps (FVG)\n"
                                "- Liquidity sweeps or stop hunts\n"
                                "Then give your SMC-based trade idea with:\n"
                                "- Market bias (Bullish or Bearish)\n"
                                "- Entry price (where to enter)\n"
                                "- Stop Loss (protective level)\n"
                                "- Take Profit (logical target)\n"
                                "- Pattern name (e.g., OB retest after CHoCH)\n"
                                "- Strategy (must say \"SMC\")\n"
                                "- Confidence from 0 to 100\n"
                                "- Commentary: a short explanation of why this trade setup is valid.\n"
                                "\nFormat your reply STRICTLY as JSON like this:\n"
                                "{\"strategy\": \"SMC\", \"signal\": \"BUY\", \"bias\": \"Bullish\", \"pattern\": \"CHoCH + OB Retest\", \"entry\": 2315.55, \"stopLoss\": 2301.25, \"takeProfit\": 2348.95, \"confidence\": 91.2, \"commentary\": \"Price swept liquidity then broke structure and retested a demand OB.\"}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=1200
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
        return AnalysisResult(strategy="SMC", signal="", bias="", pattern="", entry=0, stopLoss=0, takeProfit=0, confidence=0, commentary="")
