import os
import io
import openai
import base64
import json
import re
from fastapi import FastAPI, File, UploadFile, Form
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
    recommendedSize: str = "N/A"
    assetType: str = "Unknown"

class FullAnalysis(BaseModel):
    results: list[StrategyResult]
    superTrade: bool
    topPick: StrategyResult | None

@app.post("/analyze", response_model=FullAnalysis)
async def analyze_chart(
    file: UploadFile = File(...),
    portfolioSize: float = Form(10000),
    riskTolerance: str = Form("moderate")
):
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
                temperature=0.4,
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

                    # Basic asset type detection
                    lower_commentary = json_data.get("commentary", "").lower()
                    if any(x in lower_commentary for x in ["eurusd", "gbpusd", "usd", "pip"]):
                        asset_type = "forex"
                    elif "gold" in lower_commentary or "xau" in lower_commentary:
                        asset_type = "gold"
                    elif "btc" in lower_commentary or "crypto" in lower_commentary:
                        asset_type = "crypto"
                    elif any(x in lower_commentary for x in ["nasdaq", "s&p", "dow"]):
                        asset_type = "indices"
                    elif any(x in lower_commentary for x in ["stock", "share"]):
                        asset_type = "stock"
                    else:
                        asset_type = "Unknown"

                    json_data["assetType"] = asset_type

                    # Calculate risk
                    result = StrategyResult(**json_data)
                    risk_percent = {"low": 0.005, "moderate": 0.01, "high": 0.02}.get(riskTolerance.lower(), 0.01)
                    risk_amount = portfolioSize * risk_percent
                    stop_distance = abs(result.entry - result.stopLoss)

                    if stop_distance > 0:
                        if asset_type == "forex":
                            pip_value = 10
                            lots = risk_amount / (stop_distance / 0.0001 * pip_value)
                            result.recommendedSize = f"{lots:.2f} lots"
                        elif asset_type == "gold":
                            contracts = risk_amount / (stop_distance * 100)
                            result.recommendedSize = f"{contracts:.2f} contracts"
                        elif asset_type == "crypto":
                            coins = risk_amount / stop_distance
                            result.recommendedSize = f"{coins:.4f} units"
                        elif asset_type == "stock":
                            shares = risk_amount / stop_distance
                            result.recommendedSize = f"{int(shares)} shares"
                        elif asset_type == "indices":
                            units = risk_amount / stop_distance
                            result.recommendedSize = f"{units:.2f} contracts"
                        else:
                            units = risk_amount / stop_distance
                            result.recommendedSize = f"{units:.2f} units"

                    results.append(result)
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
        "  // Estimate this based on clarity, confluence, and strength of the pattern.\n"
        "  \"tradeType\": \"Scalp, Intraday, or Swing\",\n"
        "  \"commentary\": \"Write a detailed and insightful explanation using trading logic, structure, levels, risk-reward, and confirmation.\"\n"
        "}\n\n"
        "⚠️ Return only one flat JSON object. No nested data, no text before or after."
    )

    strategy_prompts = {
        "SMC": "You're an expert in Smart Money Concepts. Identify a CHoCH, BOS or OB Retest with precision.",
        "Breakout": "You're a breakout specialist. Spot range or trendline breaks, with confirmation if retested.",
        "Fibonacci": "You're a Fibonacci strategy pro. Identify a clean bounce or rejection at the 0.618 or 0.786 levels. Specify how the level aligns with structure.",
        "PriceAction": "You're a price action expert. Identify pin bars, engulfing candles, or market structure breaks with clean context.",
        "Reversal": "You're a reversal trader. Look for divergence, overextensions, and clear rejection patterns at logical reversal zones.",
        "Trendline": "Spot clean trendline touches, breaks, and retests. Clearly specify slope and reaction.",
        "LiquiditySweep": "Detect liquidity grabs or stop hunts followed by reversal signals.",
        "SupportResistance": "Analyze price reaction at horizontal levels. Identify rejection, fakeouts, or bounces.",
        "Scalping": "You're a scalping specialist. Look for sharp momentum shifts, microstructure breaks, and quick entries.",
        "OrderBlock": "You're an order block specialist. Identify recent OB formations and retests with confirmation."
    }

    intro = strategy_prompts.get(strategy, "")
    return intro + "\n" + schema.replace("{strategy_name}", strategy)
