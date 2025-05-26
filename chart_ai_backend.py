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
from datetime import datetime
from news_events_utils import get_recent_news, scrape_forex_factory

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

def generate_news_context():
    date_today = datetime.utcnow().strftime('%Y-%m-%d')
    news = get_recent_news("US")
    calendar = scrape_forex_factory()

    news_block = "\n".join(f"- {n['title'][:100]}" for n in news)
    econ_block = "\n".join(
        f"- {e['event']} (Impact: {e.get('impact', 'Unknown')}) at {e.get('time', '?')} [{e.get('currency', '?')}]"
        for e in calendar
    )

    return f"Economic Events (as of {date_today}):\n{econ_block}\n\nRecent News Headlines:\n{news_block}"

def generate_prompt(strategy: str, news_context: str, language: str = "en") -> str:
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
        "  \"tradeType\": \"Scalp, Intraday, or Swing\",\n"
        "  \"commentary\": \"Write a detailed and insightful explanation using trading logic, structure, levels, risk-reward, confirmation, and macro context.\"\n"
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
    full_prompt = f"{news_context}\n\n{intro}\n" + schema.replace("{strategy_name}", strategy)

    if language == "fr":
        return f"Réponds en français.\n{full_prompt}"
    elif language == "es":
        return f"Responde en español.\n{full_prompt}"
    return full_prompt
