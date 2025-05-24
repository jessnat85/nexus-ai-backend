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
    if strategy == "SMC":
        return (
            "You are an SMC trading expert. Analyze this chart for a Smart Money Concept setup."
            " Return only a JSON like this: {\"strategy\":\"SMC\", \"signal\":\"BUY\", \"bias\":\"Bullish\", \"pattern\":\"CHoCH + OB Retest\", \"entry\": 4321.50, \"stopLoss\": 4308.00, \"takeProfit\": 4375.00, \"confidence\": 88, \"commentary\": \"CHoCH occurred after a sweep, followed by OB retest.\" }"
        )
    elif strategy == "Breakout":
        return (
            "You are a trading expert. Analyze this chart for a breakout setup."
            " Return only a JSON like this: {\"strategy\":\"Breakout\", \"signal\":\"SELL\", \"bias\":\"Bearish\", \"pattern\":\"Range Break + Retest\", \"entry\": 4025.00, \"stopLoss\": 4040.00, \"takeProfit\": 3980.00, \"confidence\": 82, \"commentary\": \"Price broke range low and retested prior support as resistance.\" }"
        )
    else:
        return ""
