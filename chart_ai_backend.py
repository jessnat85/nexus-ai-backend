import os
import io
import openai
import base64
import json
import re
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

    strategies = ["SMC", "Breakout", "Fibonacci"]
    results = []

    for strategy in strategies:
        try:
            prompt = generate_prompt(strategy)
            response = openai.chat.completions.create(
                model="gpt-4o",
                temperature=0,
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
            raw = response.choices[0].message.content
            print(f"Raw GPT output ({strategy}):\n", raw)
            match = re.search(r'{.*}', raw, re.DOTALL)
            if match:
                json_data = json.loads(match.group())
                results.append(StrategyResult(**json_data))
            else:
                raise ValueError("No valid JSON object found")
        except Exception as e:
            print(f"Error in {strategy} analysis: {str(e)}")
            continue

    super_trade = False
    if len(results) >= 2:
        first_signal = results[0].signal
        if all(r.signal == first_signal for r in results):
            super_trade = True

    return FullAnalysis(results=results, superTrade=super_trade)

def generate_prompt(strategy: str) -> str:
    float_instructions = (
        " All numeric fields (entry, stopLoss, takeProfit, confidence) must be valid floats without quotes or commas."
        " Confidence must be a float between 0 and 100."
    )
    if strategy == "SMC":
        return (
            "You are an expert in Smart Money Concept trading. Analyze this chart for a valid SMC setup."
            " Focus on CHoCH, BOS, and order blocks. Ignore overlay text like 'BUY' or 'SELL'."
            " Only return a JSON with: strategy, signal, bias, pattern, entry, stopLoss, takeProfit, confidence, commentary."
            + float_instructions
        )
    elif strategy == "Breakout":
        return (
            "You are a trading expert specialized in breakout strategies. Analyze this chart for consolidation or trendline breakouts."
            " Confirm breakout with momentum and optional retest. Ignore chart overlay text."
            " Return only JSON: strategy, signal, bias, pattern, entry, stopLoss, takeProfit, confidence, commentary."
            + float_instructions
        )
    elif strategy == "Fibonacci":
        return (
            "You are a professional trader using Fibonacci retracements. Analyze this chart to detect price pulling back to levels like 0.618 or 0.786."
            " Identify if there's a bounce or reversal from those levels, ideally with confluence. Ignore visible BUY/SELL text."
            " Return only JSON: strategy, signal, bias, pattern, entry, stopLoss, takeProfit, confidence, commentary."
            + float_instructions
        )
    else:
        return ""
