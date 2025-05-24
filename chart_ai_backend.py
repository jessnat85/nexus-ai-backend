import os
import openai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64

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
    # Read and encode the image
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_uri = f"data:image/png;base64,{base64_image}"

    # Prompt for analysis
    system_prompt = (
        "You are a professional trading assistant. A user uploads a chart screenshot, and you must interpret it."
        " Identify the trading strategy (like SMC, Breakout, Reversal), signal (Buy/Sell), bias (Bullish/Bearish), pattern (e.g., Order Block, Liquidity Sweep),"
        " and give estimated entry, stop loss (SL), and take profit (TP) levels if visible. Return a confidence score out of 100%."
        " Format response strictly in this JSON:
        {\"strategy\": \"\", \"signal\": \"\", \"bias\": \"\", \"pattern\": \"\", \"entry\": 0.0, \"stopLoss\": 0.0, \"takeProfit\": 0.0, \"confidence\": 0.0}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": "Please analyze this chart and return results in JSON format."},
                    ],
                },
            ],
            max_tokens=1000,
        )

        # Extract and parse the JSON response
        content = response.choices[0].message.content
        print("GPT raw output:", content)
        import json
        parsed = json.loads(content)

        return AnalysisResult(**parsed)

    except Exception as e:
        print("Error during GPT Vision analysis:", str(e))
        return AnalysisResult(
            strategy="N/A",
            signal="",
            bias="",
            pattern="",
            entry=0.0,
            stopLoss=0.0,
            takeProfit=0.0,
            confidence=0.0,
        )
