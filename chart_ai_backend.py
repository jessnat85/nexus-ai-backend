# chart_ai_backend.py - FastAPI backend for Nexus AI (dynamic version)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    signal: str
    bias: str
    pattern: str
    entry: float
    stopLoss: float
    takeProfit: float
    confidence: Optional[int] = 90

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_chart(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Dynamic signal simulation
    patterns = ["Fakeout + Bullish Engulfing", "Double Bottom", "Order Block Retest", "Liquidity Sweep + Reversal"]
    signals = ["BUY", "SELL"]
    biases = ["Bullish", "Bearish"]

    signal = random.choice(signals)
    bias = random.choice(biases)
    pattern = random.choice(patterns)
    entry = round(random.uniform(2300, 2330), 2)
    stopLoss = round(entry - random.uniform(5, 10), 2)
    takeProfit = round(entry + random.uniform(10, 20), 2)
    confidence = random.randint(85, 99)

    return AnalysisResult(
        signal=signal,
        bias=bias,
        pattern=pattern,
        entry=entry,
        stopLoss=stopLoss,
        takeProfit=takeProfit,
        confidence=confidence
    )
