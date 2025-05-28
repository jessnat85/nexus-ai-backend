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

# Instrument metadata map
INSTRUMENT_MAP = {
    "MNQ": {"assetType": "Indices", "pointValue": 2, "tickSize": 0.25, "unitLabel": "micro contracts"},
    "NQ": {"assetType": "Indices", "pointValue": 20, "tickSize": 0.25, "unitLabel": "contracts"},
    "ES": {"assetType": "Indices", "pointValue": 50, "tickSize": 0.25, "unitLabel": "contracts"},
    "MES": {"assetType": "Indices", "pointValue": 5, "tickSize": 0.25, "unitLabel": "micro contracts"},
    "XAUUSD": {"assetType": "Gold", "pointValue": 1, "tickSize": 0.1, "unitLabel": "ounces"},
    "EURUSD": {"assetType": "Forex", "pipValue": 10, "pipSize": 0.0001, "unitLabel": "lots"},
    "USDJPY": {"assetType": "Forex", "pipValue": 10, "pipSize": 0.01, "unitLabel": "lots"},
    "BTCUSD": {"assetType": "Crypto", "pointValue": 1, "tickSize": 1, "unitLabel": "BTC"},
    "ETHUSD": {"assetType": "Crypto", "pointValue": 1, "tickSize": 1, "unitLabel": "ETH"},
}

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
    conflictCommentary: str | None = None

def detect_symbol_and_metadata(image_b64):
    prompt = (
        "You're an AI that reads trading charts. Extract the symbol name, asset type, point value, tick size, and unit label.\n"
        "Return this as JSON:\n"
        "{\n  \"symbol\": \"MNQ\",\n  \"assetType\": \"Indices\",\n  \"pointValue\": 2,\n  \"tickSize\": 0.25,\n  \"unitLabel\": \"micro contracts\"\n}"
    )
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]
            }
        ]
    )
    try:
        match = re.search(r'{.*}', response.choices[0].message.content, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except Exception as e:
        print(f"Symbol detection failed: {e}")
        return {}

def calculate_recommended_size(entry, stopLoss, portfolioSize, riskTolerance, meta):
    risk_pct = {"low": 0.005, "moderate": 0.01, "high": 0.02}.get(riskTolerance.lower(), 0.01)
    risk_amount = portfolioSize * risk_pct
    sl_points = abs(entry - stopLoss)

    if sl_points == 0:
        return "N/A"

    if "pipSize" in meta:
        pips = sl_points / meta["pipSize"]
        dollar_risk = pips * meta["pipValue"]
        lots = risk_amount / dollar_risk
        return f"{lots:.2f} {meta['unitLabel']}"
    else:
        dollar_risk = sl_points * meta["pointValue"]
        size = risk_amount / dollar_risk
        return f"{size:.2f} {meta['unitLabel']}"

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

    symbol_meta = detect_symbol_and_metadata(img_b64)
    fallback_symbol = re.sub(r'[\d\-]+', '', symbol_meta.get("symbol", "")).upper()
    meta = INSTRUMENT_MAP.get(fallback_symbol, {
        "assetType": symbol_meta.get("assetType", "Unknown"),
        "pointValue": symbol_meta.get("pointValue", 1),
        "tickSize": symbol_meta.get("tickSize", 0.1),
        "unitLabel": symbol_meta.get("unitLabel", "units")
    })

    strategies = ["SMC", "Breakout", "Fibonacci", "PriceAction", "Reversal",
                  "Trendline", "LiquiditySweep", "SupportResistance", "Scalping", "SupplyDemand", "NexusPulse"]
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
                        ]
                    }
                ],
                max_tokens=1000
            )
            raw = response.choices[0].message.content
            match = re.search(r'{.*}', raw, re.DOTALL)
            if not match:
                continue
            json_data = json.loads(match.group())
            json_data["strategy"] = strategy
            json_data["assetType"] = meta["assetType"]
            if "entry" in json_data and "stopLoss" in json_data:
                json_data["recommendedSize"] = calculate_recommended_size(
                    json_data["entry"], json_data["stopLoss"], portfolioSize, riskTolerance, meta
                )
            results.append(StrategyResult(**json_data))
        except Exception as e:
            print(f"Error in {strategy}: {e}")
            continue

    super_trade = False
    top_pick = None
    conflict_commentary = None

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
        top_pick = max(results, key=lambda r: (r.confidence, (r.takeProfit - r.entry) / max(0.01, abs(r.entry - r.stopLoss))))

    if any(r.signal != top_pick.signal for r in results):
        conflict_commentary = "⚠️ Conflicting trade signals detected. Strategies are not fully aligned. Proceed with caution."

    return FullAnalysis(results=results, superTrade=super_trade, topPick=top_pick, conflictCommentary=conflict_commentary)

def generate_prompt(strategy: str) -> str:
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

        "SupplyDemand": "You are a trading assistant specialized in Supply and Demand analysis.\n\nInstructions:\n1. Identify clear supply (rally-base-drop) and demand (drop-base-rally) zones.\n2. Focus on zones with strong imbalance followed by a large move.\n3. Confirm the zone is fresh and check for price return with a rejection pattern.\n\nIf no reaction or fresh zone exists, return:\n{\"superTrade\": false, \"commentary\": \"No strong reaction at any supply or demand zone.\"}",

        "NexusPulse": "You are a general trading assistant providing high-level insight when no specific strategy applies.\n\nInstructions:\n1. Analyze trend, price structure (HH/HL or LL/LH), support/resistance, and recent volatility.\n2. Identify if the market is trending, ranging, or reversing.\n3. Offer a possible trade setup if there's logic (entry, SL, TP, bias), even if confluence is low.\n4. Clearly communicate uncertainty or market caution if applicable.\n\nUse looser thresholds and provide broader guidance.\nIf absolutely no trade idea is viable, return:\n{\"commentary\": \"No reasonable trade idea found given current market conditions.\"}"
    }

    fallback = (
        "\n\nIf no valid setup is found with strict criteria, slightly relax the conditions "
        "and attempt to generate the best possible trade idea. Use a confidence score between 60 and 70 "
        "and clearly explain any uncertainty or weakness in the commentary."
    )

    schema = (
        "\n\nRespond only with a JSON object using this exact schema:\n"
        "{\n"
        f"  \"strategy\": \"{strategy}\",\n"
        "  \"signal\": \"Buy or Sell\",\n"
        "  \"bias\": \"Bullish or Bearish\",\n"
        "  \"pattern\": \"Describe the key pattern you used\",\n"
        "  \"entry\": float,\n"
        "  \"stopLoss\": float,\n"
        "  \"takeProfit\": float,\n"
        "  \"confidence\": float (0 to 100),\n"
        "  \"tradeType\": \"Scalp, Intraday, or Swing\",\n"
        "  \"commentary\": \"Detailed explanation using logic, structure, risk-reward, confluence.\"\n"
        "}\n"
        "⚠️ Return only a single flat JSON object. Do not return any text before or after."
    )

    prompt_intro = strategy_prompts.get(strategy, "You are a trading assistant.")
    return prompt_intro + fallback + schema

    fallback = (
        "\n\nIf no valid setup is found with strict criteria, slightly relax the conditions "
        "and attempt to generate the best possible trade idea. Use a confidence score between 60 and 70 "
        "and clearly explain any uncertainty or weakness in the commentary."
    )

    schema = (
        "\n\nRespond only with a JSON object using this exact schema:\n"
        "{\n"
        f"  \"strategy\": \"{strategy}\",\n"
        "  \"signal\": \"Buy or Sell\",\n"
        "  \"bias\": \"Bullish or Bearish\",\n"
        "  \"pattern\": \"Describe the key pattern you used\",\n"
        "  \"entry\": float,\n"
        "  \"stopLoss\": float,\n"
        "  \"takeProfit\": float,\n"
        "  \"confidence\": float (0 to 100),\n"
        "  \"tradeType\": \"Scalp, Intraday, or Swing\",\n"
        "  \"commentary\": \"Detailed explanation using logic, structure, risk-reward, confluence.\"\n"
        "}\n"
        "⚠️ Return only a single flat JSON object. Do not return any text before or after."
    )

    prompt_intro = strategy_prompts.get(strategy, "You are a trading assistant.")
    return prompt_intro + fallback + schema
