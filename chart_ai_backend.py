from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
import pytesseract
import numpy as np
import cv2
import re
import io
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
    strategy: str
    signal: str
    bias: str
    pattern: str
    entry: float
    stopLoss: float
    takeProfit: float
    confidence: float


def extract_prices_from_image(image: Image.Image) -> List[float]:
    image = np.array(image)
    height, width = image.shape[:2]
    
    # Crop to right edge where y-axis prices are
    crop = image[0:height, int(width * 0.88):width]

    # Preprocess
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # OCR config
    raw_text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789.')
    print("RAW OCR TEXT:", raw_text)

    matches = re.findall(r'\d{2,7}\.\d{2}', raw_text)
    prices = sorted(set([float(m) for m in matches if float(m) > 10.0]))
    print("MATCHED PRICES:", prices)

    return prices


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_chart(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    prices = extract_prices_from_image(image)

    # Simulate AI output for demo purpose
    strategy = random.choice(["SMC", "Breakout", "Reversal", "Trend Following"])
    signal = random.choice(["BUY", "SELL"])
    bias = "Bullish" if signal == "BUY" else "Bearish"
    pattern = random.choice(["Order Block", "Breakout", "Liquidity Sweep", "Engulfing", "FVG"])
    confidence = round(random.uniform(60, 99), 2)

    # Generate fake price levels from OCR data
    if len(prices) >= 3:
        entry = prices[len(prices) // 2]
        stopLoss = prices[0]
        takeProfit = prices[-1]
    else:
        entry = stopLoss = takeProfit = 0.0

    return AnalysisResult(
        strategy=strategy,
        signal=signal,
        bias=bias,
        pattern=pattern,
        entry=round(entry, 2),
        stopLoss=round(stopLoss, 2),
        takeProfit=round(takeProfit, 2),
        confidence=confidence
    )
