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
    tradeType: str

class FullAnalysis(BaseModel):
    results: list[StrategyResult]
    superTrade: bool
    topPick: StrategyResult | None

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
            print(f"Raw GPT output ({strategy}):\n{raw}")

            match = re.search(r'{.*}', raw, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group())
                    json_data["strategy"] = strategy
                    results.append(StrategyResult(**json_data))
                except Exception as json_err:
                    print(f"⚠️ Skipping {strategy} - JSON parse error: {json_err}")
                    continue
            else:
                print(f"⚠️ Skipping {strategy} - No valid JSON object found")
                continue

        except Exception as e:
            print(f"⚠️ Error during {strategy} analysis: {str(e)}")
            continue

    super_trade = False
    top_pick = None

    if len(results) >= 3:
        # Group by signal
        buy_signals = [r for r in results if r.signal == "Buy"]
        sell_signals = [r for r in results if r.signal == "Sell"]

        def check_confluence(group):
            if len(group) < 3:
                return False
            avg_conf = sum(r.confidence for r in group) / len(group)
            same_bias = all(r.bias == group[0].bias for r in group)
            rr_ok = all(((r.takeProfit - r.entry) / abs(r.entry - r.stopLoss)) >= 1.5 for r in group)
            return avg_conf >= 78 and same_bias and rr_ok

        if check_confluence(buy_signals) or check_confluence(sell_signals):
            super_trade = True

    if results:
        top_pick = max(results, key=lambda r: r.confidence)

    return FullAnalysis(results=results, superTrade=super_trade, topPick=top_pick)

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
        "  \"tradeType\": \"Scalp, Intraday, Swing\",\n"
        "  \"commentary\": \"Explain the rationale behind the trade setup.\"\n"
        "}\n\n"
        "⚠️ You must not return multiple setups or nested keys. Only one flat JSON object as shown. "
        "All numbers must be valid floats (no commas or quotes)."
    )

    strategy_prompts = {
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

    prompt_intro = strategy_prompts.get(strategy, "")
    return prompt_intro + schema.replace("{strategy_name}", strategy)
