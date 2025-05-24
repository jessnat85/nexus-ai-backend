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
    riskReward: float
    tradeType: str
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

    strategies = [
        "SMC", "Breakout", "Fibonacci", "PriceAction", "Reversal",
        "Trendline", "LiquiditySweep", "SupportResistance", "Scalping", "OrderBlock"
    ]
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
                json_data["strategy"] = strategy
                json_data["riskReward"] = calculate_rr(
                    json_data["entry"],
                    json_data["stopLoss"],
                    json_data["takeProfit"],
                    json_data["signal"]
                )
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

def calculate_rr(entry, stop, target, signal):
    try:
        if signal.lower() == "buy":
            return round((target - entry) / (entry - stop), 2)
        else:
            return round((entry - target) / (stop - entry), 2)
    except ZeroDivisionError:
        return 0.0

def generate_prompt(strategy: str) -> str:
    schema = (
        "Respond only with a JSON object using this exact schema:\n"
        "{\n"
        "  \"strategy\": \"{strategy_name}\",\n"
        "  \"signal\": \"Buy or Sell\",\n"
        "  \"bias\": \"Bullish or Bearish\",\n"
        "  \"pattern\": \"Describe the key pattern you used\",\n"
        "  \"entry\": float,\n"
        "  \"stopLoss\": float,\n"
        "  \"takeProfit\": float,\n"
        "  \"confidence\": float (0 to 100),\n"
        "  // Estimate this based on clarity, confluence, and strength of the pattern. Do NOT use vague terms like 'High'.\n"
        "  \"riskReward\": float,\n"
        "  \"tradeType\": \"Reversal, Breakout, Pullback, etc.\",\n"
        "  \"commentary\": \"Explain the rationale behind the trade setup.\"\n"
        "}\n\n"
        "⚠️ You must not return multiple setups or nested keys. Only one flat JSON object as shown."
    )
    base = {
        "SMC": "You're a Smart Money Concept (SMC) expert. Analyze this chart for one high-probability SMC trade setup based on CHoCH, BOS, or OB retest. ",
        "Breakout": "You're a breakout strategy expert. Identify range breaks or trendline breaks and retests. ",
        "Fibonacci": "You're a Fibonacci retracement expert. Detect reactions to 0.618/0.786 retracements. ",
        "PriceAction": "You're a price action expert. Look for engulfing candles, pin bars at key levels, and structure shifts. ",
        "Reversal": "Analyze this chart for reversal setups: divergence, candle patterns, or exhaustion at levels. ",
        "Trendline": "You're a trendline expert. Look for touches and breaks of major trendlines with retests. ",
        "LiquiditySweep": "Detect fakeouts or liquidity grabs followed by reversals. Look for price wicks and reaction. ",
        "SupportResistance": "Look for bounces or breaks from horizontal support/resistance levels. Ignore indicators. ",
        "Scalping": "You're a scalper. Look for microstructure shifts and fast momentum entries. ",
        "OrderBlock": "You specialize in order blocks. Identify OB formation and retests with confirmation. ",
    }
    return base.get(strategy, "") + schema.replace("{strategy_name}", strategy)
