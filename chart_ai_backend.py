import os
import io
import openai
import base64
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StrategyResult(BaseModel):
    strategy: str
    signal: str
    bias: str
    pattern: str
    entry: float
    stopLoss: float
    takeProfit: float
    confidence: float
    commentary: str

class FullAnalysis(BaseModel):
    results: list[StrategyResult]
    superTrade: bool

@app.post("/analyze", response_model=FullAnalysis)
async def analyze_chart(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    strategies = ["SMC", "Breakout"]
    results = []

    for strategy in strategies:
        try:
            prompt = generate_prompt(strategy)
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    }
                ],
                max_tokens=1000
            )
            content = response.choices[0].message.content
            data = json.loads(content.strip("`\n"))
            results.append(StrategyResult(**data))
        except Exception as e:
            print(f"Error in {strategy} analysis: {str(e)}")
            continue

    # Determine if multiple strategies align
    super_trade = False
    if len(results) >= 2:
        first_signal = results[0].signal
        if all(r.signal == first_signal for r in results):
            super_trade = True

    return FullAnalysis(results=results, superTrade=super_trade)

def generate_prompt(strategy: str) -> str:
    if strategy == "SMC":
        return (
            "You are an SMC trading expert. Analyze this chart for an SMC setup."
            " Return JSON with keys: strategy, signal, bias, pattern, entry, stopLoss, takeProfit, confidence, commentary."
            " Use this format: {\"strategy\":\"SMC\", ...}"
        )
    elif strategy == "Breakout":
        return (
            "You are a trading expert. Analyze this chart for a breakout setup."
            " Identify range or trendline breaks. Return JSON with keys: strategy, signal, bias, pattern, entry, stopLoss, takeProfit, confidence, commentary."
            " Use this format: {\"strategy\":\"Breakout\", ...}"
        )
    else:
        return ""
