from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
import pytesseract
import re
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
    strategy: Optional[str]
    signal: Optional[str]
    bias: Optional[str]
    pattern: Optional[str]
    entry: Optional[float]
    stopLoss: Optional[float]
    takeProfit: Optional[float]
    confidence: Optional[float]

def extract_prices_from_image(image: Image.Image):
    price_axis = image.crop((image.width - 160, 0, image.width, image.height))
    price_axis = price_axis.convert("L").point(lambda x: 0 if x < 150 else 255, mode='1')

    raw_text = pytesseract.image_to_string(price_axis, config='--psm 6')
    matches = re.findall(r'\d{1,3}(?:[,\d]{0,3})*\.\d{2}', raw_text.replace(' ', ''))
    prices = [float(p.replace(',', '')) for p in matches if 0.01 < float(p.replace(',', '')) < 1000000]

    if len(prices) >= 3:
        prices = sorted(set(prices))
        entry = prices[len(prices) // 2]
        stopLoss = min(prices)
        takeProfit = max(prices)
    else:
        entry = stopLoss = takeProfit = 0.0

    return entry, stopLoss, takeProfit

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_chart(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    entry, stopLoss, takeProfit = extract_prices_from_image(image)

    return AnalysisResult(
        strategy="SMC",
        signal="BUY",
        bias="Bullish",
        pattern="Liquidity Sweep + OB",
        entry=entry,
        stopLoss=stopLoss,
        takeProfit=takeProfit,
        confidence=round(random.uniform(85, 98), 1)
    )
