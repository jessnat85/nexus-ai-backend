import os
import io
import base64
import json
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

class Zone(BaseModel):
    type: str
    price: float
    status: str

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
    structure: Optional[str] = None
    riskReward: Optional[float] = None
    zones: Optional[List[Zone]] = []

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
                                "You're a Smart Money Concepts (SMC) trading expert. Analyze this chart and return the following strictly as JSON:\n"
                                "- strategy (always 'SMC')\n"
                                "- signal: BUY or SELL\n"
                                "- bias: Bullish or Bearish\n"
                                "- pattern: e.g., 'CHoCH + OB Retest'\n"
                                "- entry, stopLoss, takeProfit (float)\n"
                                "- confidence: 0-100\n"
                                "- commentary: explain the setup\n"
                                "- structure: e.g., 'LL → CHoCH → HH'\n"
                                "- riskReward: R:R estimate (float)\n"
                                "- zones: array of detected order blocks with type (Supply/Demand), price (float), and status (fresh/mitigated)\n"
                                "\nExample JSON format:\n"
                                "{\n  \"strategy\": \"SMC\",\n  \"signal\": \"SELL\",\n  \"bias\": \"Bearish\",\n  \"pattern\": \"BOS + OB Retest\",\n  \"entry\": 2565.00,\n  \"stopLoss\": 2580.00,\n  \"takeProfit\": 2510.00,\n  \"confidence\": 88.5,\n  \"commentary\": \"Price broke structure and retested supply OB after sweep.\",\n  \"structure\": \"LL → CHoCH → LH\",\n  \"riskReward\": 2.3,\n  \"zones\": [\n    { \"type\": \"Supply\", \"price\": 2575.25, \"status\": \"fresh\" },\n    { \"type\": \"Demand\", \"price\": 2502.75, \"status\": \"mitigated\" }\n  ]\n}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=1600
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
        return AnalysisResult(
            strategy="SMC",
            signal="",
            bias="",
            pattern="",
            entry=0,
            stopLoss=0,
            takeProfit=0,
            confidence=0,
            commentary="",
            structure="",
            riskReward=0,
            zones=[]
        )
