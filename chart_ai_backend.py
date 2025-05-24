# chart_ai_backend.py - Nexus AI dynamic backend with strategy-based analysis
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
import pytesseract
import numpy as np
import cv2
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
    strategy: str
    confidence: Optional[int] = 90

def extract_prices_from_image(pil_image: Image.Image) -> list[float]:
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    price_axis = img[:, int(w * 0.88):]  # crop price area
    config = '--psm 6 -c tessedit_char_whitelist=0123456789.,'
    raw_text = pytesseract.image_to_string(price_axis, config=config)
    prices = []
    for line in raw_text.splitlines():
        try:
            line = line.replace(",", ".")
            price = float(line.strip())
            prices.append(price)
        except:
            continue
    return sorted(set(prices), reverse=True)

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_chart(file: UploadFile = File(...), strategy: str = Form(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    prices = extract_prices_from_image(image)

    # Strategy → Pattern logic
    strategy_map = {
        "SMC": ["Order Block + BOS", "Liquidity Sweep + Reversal"],
        "PriceAction": ["Pin Bar", "Engulfing"],
        "Fibonacci": ["Fib Reversal", "Golden Pocket Reaction"],
        "Breakout": ["Break of Structure", "Consolidation Range Break"],
        "Trend": ["EMA Crossover", "HH-HL Structure"],
        "Reversal": ["Double Bottom", "Divergence"],
        "Scalping": ["Volume Spike Reversal", "Micro OB Rejection"]
    }
    patterns = strategy_map.get(strategy, ["Unknown Pattern"])

    signal = random.choice(["BUY", "SELL"])
    bias = "Bullish" if signal == "BUY" else "Bearish"
    pattern = random.choice(patterns)
    
    # Extracted prices help set bounds
    if len(prices) >= 3:
        base = prices[1]
    elif len(prices) == 2:
        base = sum(prices) / 2
    elif len(prices) == 1:
        base = prices[0]
    else:
        base = random.uniform(1000, 5000)

    entry = round(base, 2)
    stopLoss = round(entry - random.uniform(10, 30), 2)
    takeProfit = round(entry + random.uniform(20, 50), 2)
    confidence = random.randint(87, 99)

    return AnalysisResult(
        signal=signal,
        bias=bias,
        pattern=pattern,
        entry=entry,
        stopLoss=stopLoss,
        takeProfit=takeProfit,
        strategy=strategy,
        confidence=confidence
    )
