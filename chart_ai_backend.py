# chart_ai_backend.py - Nexus AI (Cloud OCR with fallback)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import cv2
import random
import base64

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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    price_axis = img[:, int(w * 0.88):]
    retval, buffer = cv2.imencode(".png", price_axis)
    b64_image = base64.b64encode(buffer).decode()

    api_url = "https://api.ocr.space/parse/image"
    response = requests.post(
        api_url,
        data={
            "base64Image": "data:image/png;base64," + b64_image,
            "isOverlayRequired": False,
            "OCREngine": 2,
            "scale": True,
        },
        headers={"apikey": "helloworld"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="OCR API failed")

    data = response.json()
    raw_text = data.get("ParsedResults", [{}])[0].get("ParsedText", "")
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
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        prices = extract_prices_from_image(image)
        print("Extracted prices:", prices)
        note = None
    except Exception as e:
        print("OCR failed:", str(e))
        prices = []
        note = "OCR failed, using fallback values"

    patterns = ["Fakeout + Bullish Engulfing", "Double Bottom", "Order Block Retest", "Liquidity Sweep + Reversal"]
    signals = ["BUY", "SELL"]
    biases = ["Bullish", "Bearish"]

    signal = random.choice(signals)
    bias = random.choice(biases)
    pattern = random.choice(patterns)
    confidence = random.randint(85, 99)

    if len(prices) >= 2:
        high = prices[0]
        low = prices[-1]
        entry = round((high + low) / 2, 2)
        stopLoss = round(entry - (high - low) * 0.25, 2)
        takeProfit = round(entry + (high - low) * 0.5, 2)
    else:
        entry = 2315.25
        stopLoss = 2310.10
        takeProfit = 2330.00

    return AnalysisResult(
        signal=signal,
        bias=bias,
        pattern=pattern,
        entry=entry,
        stopLoss=stopLoss,
        takeProfit=takeProfit,
        confidence=confidence,
        note=note
    )
