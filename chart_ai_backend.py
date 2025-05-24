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
        "  \"commentary\": \"Explain the rationale behind the trade setup.\"\n"
        "}\n\n"
        "⚠️ You must not return multiple setups or nested keys. Only one flat JSON object as shown. "
        "All numbers must be valid floats (no commas or quotes)."
    )
    if strategy == "SMC":
        return "You're a Smart Money Concept (SMC) expert. Analyze this chart for one high-probability SMC trade setup based on CHoCH, BOS, or OB retest. " + schema.replace("{strategy_name}", "SMC")
    if strategy == "Breakout":
        return "You're a breakout strategy expert. Identify range breaks or trendline breaks and retests. " + schema.replace("{strategy_name}", "Breakout")
    if strategy == "Fibonacci":
        return "You're a Fibonacci retracement expert. Detect reactions to 0.618/0.786 retracements. " + schema.replace("{strategy_name}", "Fibonacci")
    if strategy == "PriceAction":
        return "You're a price action expert. Look for engulfing candles, pin bars at key levels, and structure shifts. " + schema.replace("{strategy_name}", "PriceAction")
    if strategy == "Reversal":
        return "Analyze this chart for reversal setups: divergence, candle patterns, or exhaustion at levels. " + schema.replace("{strategy_name}", "Reversal")
    if strategy == "Trendline":
        return "You're a trendline expert. Look for touches and breaks of major trendlines with retests. " + schema.replace("{strategy_name}", "Trendline")
    if strategy == "LiquiditySweep":
        return "Detect fakeouts or liquidity grabs followed by reversals. Look for price wicks and reaction. " + schema.replace("{strategy_name}", "LiquiditySweep")
    if strategy == "SupportResistance":
        return "Look for bounces or breaks from horizontal support/resistance levels. Ignore indicators. " + schema.replace("{strategy_name}", "SupportResistance")
    if strategy == "Scalping":
        return "You're a scalper. Look for microstructure shifts and fast momentum entries. " + schema.replace("{strategy_name}", "Scalping")
    if strategy == "OrderBlock":
        return "You specialize in order blocks. Identify OB formation and retests with confirmation. " + schema.replace("{strategy_name}", "OrderBlock")
    return ""
