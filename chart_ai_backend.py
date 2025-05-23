# chart_ai_backend.py - FastAPI backend for Nexus AI (dynamic version)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
import pytesseract
import numpy as np
import cv2
import random
def extract_prices_from_image(pil_image: Image.Image) -> list[float]:
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    price_axis = img[:, int(w * 0.88):]
    config = '--psm 6 -c tessedit_char_whitelist=0123456789.,'
    raw_text = pytesseract.image_to_string(price_axis, config=config)
    prices = []
    for line in raw_text.splitlines():
        try:
            line = line.replace(',', '.')
            price = float(line.strip())
            prices.append(price)
        except:
            continue
    return sorted(set(prices), reverse=True)

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
    prices = extract_prices_from_image(image)
    print("Extracted prices:", prices)
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
