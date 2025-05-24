from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import random

app = FastAPI()

# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    strategy: Optional[str]
    signal: Optional[str]
    bias: Optional[str]
    pattern: Optional[str]
    entry: Optional[float]
    stopLoss: Optional[float]
    takeProfit: Optional[float]
    confidence: Optional[float]

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_chart(file: UploadFile = File(...)):
    contents = await file.read()

    # Placeholder AI logic (simulate analysis)
    # You would replace this with actual image analysis code
    strategy = random.choice(["SMC", "Breakout", "Reversal", "Trend Following"])
    signal = random.choice(["BUY", "SELL"])
    bias = random.choice(["Bullish", "Bearish"])
    pattern = random.choice(["Liquidity Sweep + Reversal", "Fakeout", "Breakout", "Order Block"])
    entry = round(random.uniform(10000, 11000), 2)
    stopLoss = round(entry - random.uniform(100, 250), 2)
    takeProfit = round(entry + random.uniform(200, 500), 2)
    confidence = round(random.uniform(70, 98), 2)

    return AnalysisResult(
        strategy=strategy,
        signal=signal,
        bias=bias,
        pattern=pattern,
        entry=entry,
        stopLoss=stopLoss,
        takeProfit=takeProfit,
        confidence=confidence
    )
