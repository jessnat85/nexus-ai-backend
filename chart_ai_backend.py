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

                    if "entry" not in json_data or "signal" not in json_data:
    print(f"⚠️ Skipping {strategy} - No trade setup, only commentary.")
    continue

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
        "SMC": "You are a professional trading assistant specialized in Smart Money Concepts (SMC).\n\nInstructions:\n1. Identify market structure (Higher Highs, Lower Lows), Break of Structure (BOS), Change of Character (CHoCH), valid Order Blocks (OBs), liquidity sweeps, and Fair Value Gaps (FVG).\n2. Determine if a valid SMC setup exists.\n3. Only generate a trade if BOS is followed by a valid OB and liquidity sweep.\n\nIf no valid setup exists, return:\n{\"superTrade\": false, \"commentary\": \"No valid SMC setup found.\"}",
        "Breakout": "You are a breakout strategy analyst.\n\nInstructions:\n1. Identify consolidation zones (tight range, 3+ candles).\n2. Detect breakout candles with volume or momentum.\n3. Determine breakout direction and whether there's a valid retest.\n\nOnly suggest trades with breakout + retest or momentum continuation. If none, return:\n{\"superTrade\": false, \"commentary\": \"No valid breakout pattern found.\"}",
        "Fibonacci": "You are a trading assistant specialized in Fibonacci-based analysis.\n\nInstructions:\n1. Detect the swing high and swing low.\n2. Draw Fibonacci retracement zones (0.618, 0.5, 0.382).\n3. Check for price reaction (e.g., rejection wick or engulfing).\n4. Confirm with trend for confluence.\n\nIf no valid reaction, return:\n{\"superTrade\": false, \"commentary\": \"No valid Fibonacci reaction found.\"}",
        "PriceAction": "You are a professional trading assistant focused on price action setups.\n\nInstructions:\n1. Look for candlestick patterns like engulfing, pin bars, inside bars.\n2. Confirm at key levels with structure or S/R confluence.\n\nIf valid setup appears, return trade. Else, return:\n{\"superTrade\": false, \"commentary\": \"No strong price action signal detected.\"}",
        "Reversal": "You are an expert in identifying market reversal setups.\n\nInstructions:\n1. Look for extended trends (3+ candles in one direction).\n2. Find patterns like double tops/bottoms, head & shoulders, divergence.\n3. Confirm with rejection or engulfing candles.\n\nIf reversal structure is not clear, return:\n{\"superTrade\": false, \"commentary\": \"No strong reversal structure observed.\"}",
        "Trendline": "You specialize in analyzing trendlines.\n\nInstructions:\n1. Identify upward or downward sloping trendlines with 2+ touches.\n2. Check for bounce or breakout from trendline.\n3. Confirm with strong close or volume spike.\n\nIf none found, return:\n{\"superTrade\": false, \"commentary\": \"No valid trendline break or bounce identified.\"}",
        "LiquiditySweep": "You specialize in liquidity hunts and sweeps.\n\nInstructions:\n1. Look for equal highs/lows or stop clusters.\n2. Check if price swept those zones and reversed.\n3. Confirm with structure shift or reversal candle.\n\nIf no confirmation follows sweep, return:\n{\"superTrade\": false, \"commentary\": \"No liquidity grab followed by confirmation.\"}",
        "SupportResistance": "You are a trading assistant specialized in support and resistance trading strategies.\n\nInstructions:\n1. Identify recent key horizontal levels where price reacted at least twice (support or resistance).\n2. Analyze if price recently bounced from, rejected, or broke one of those levels.\n3. Confirm the setup using a strong candle signal (e.g., engulfing, pin bar) or clear market momentum.\n4. Ensure the trade setup has logical stop loss and take profit based on structure.\n\nOnly generate a trade setup if:\n- A support or resistance level was tested or rejected.\n- There is a confirmation candle or breakout structure.\n\nIf no valid setup exists, return:\n{\"superTrade\": false, \"commentary\": \"No valid trade near support/resistance.\"}",
        "Scalping": "You are analyzing for fast, low-risk scalping trades.\n\nInstructions:\n1. Look for micro price reactions at key zones.\n2. Confirm with candle patterns or fast rejection.\n3. Entry should have tight SL and fast TP (<1.5x RR).\n\nIf no suitable setup, return:\n{\"superTrade\": false, \"commentary\": \"No suitable scalping setup available.\"}",
        "SupplyDemand": "You are a trading assistant specialized in Supply and Demand analysis.\n\nInstructions:\n1. Identify clear supply (rally-base-drop) and demand (drop-base-rally) zones.\n2. Focus on zones with strong imbalance followed by a large move.\n3. Confirm the zone is fresh and check for price return with a rejection pattern.\n\nIf no reaction or fresh zone exists, return:\n{\"superTrade\": false, \"commentary\": \"No strong reaction at any supply or demand zone.\"}"
    }

    intro = strategy_prompts.get(strategy, "")
    return intro + "\n" + schema.replace("{strategy_name}", strategy)
