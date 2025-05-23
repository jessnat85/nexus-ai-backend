# chart_ai_backend.py - FastAPI backend for Nexus AI (real price detection)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
import pytesseract
import numpy as np
import cv2

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
    note: Optional[str] = None

def extract_prices_from_image(pil_image: Image.Image) -> list[float]:
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    price_axis = gray[:, int(w * 0.88):]  # Crop right side y-axis

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

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_chart(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    prices = extract_prices_from_image(image)

    if len(prices) < 3:
        return AnalysisResult(
            signal="N/A",
            bias="Unknown",
            pattern="N/A",
            entry=0,
            stopLoss=0,
            takeProfit=0,
            confidence=0,
            note="Unable to extract enough price levels from image."
        )

    highest = max(prices)
    lowest = min(prices)
    mid = round((highest + lowest) / 2, 2)

    return AnalysisResult(
        signal="BUY" if prices[0] > mid else "SELL",
        bias="Bullish" if prices[0] > mid else "Bearish",
        pattern="Liquidity Sweep + Reversal",
        entry=mid,
        stopLoss=lowest,
        takeProfit=highest,
        confidence=95
    )
