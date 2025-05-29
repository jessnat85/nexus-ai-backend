import os
import io
import re
import json
import base64
from datetime import datetime

import openai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from PIL import Image
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "sqlite:///./trades.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class TradeResult(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    userId = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    strategy = Column(String)
    signal = Column(String)
    bias = Column(String)
    pattern = Column(String)
    entry = Column(Float)
    stopLoss = Column(Float)
    takeProfit = Column(Float)
    takeProfit2 = Column(Float, nullable=True)              # <-- TP2 added
    confidence = Column(Float)
    tradeType = Column(String)
    recommendedSize = Column(String)
    assetType = Column(String)
    commentary = Column(String)
    isSuperTrade = Column(Boolean, default=False)
    isTopPick = Column(Boolean, default=False)


Base.metadata.create_all(bind=engine)

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
    takeProfit2: float | None = None                        # <-- TP2 optional
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


def detect_symbol_and_metadata(image_b64: str) -> dict:
    prompt = "You're an AI that reads trading charts. Extract symbol, asset type, point value, tick size, unit label in JSON."
    resp = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
    )
    try:
        match = re.search(r"{.*}", resp.choices[0].message.content, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except Exception:
        return {}


def calculate_recommended_size(entry: float, sl: float, portfolio: float, risk: str, meta: dict) -> str:
    pct = {"low": 0.005, "moderate": 0.01, "high": 0.02}.get(risk.lower(), 0.01)
    risk_amt = portfolio * pct
    dist = abs(entry - sl)
    if dist == 0:
        return "N/A"
    if "pipSize" in meta:
        pips = dist / meta["pipSize"]
        dollars = pips * meta["pipValue"]
        lots = risk_amt / dollars if dollars else 0
        return f"{lots:.2f} {meta['unitLabel']}"
    dollars = dist * meta["pointValue"]
    units = risk_amt / dollars if dollars else 0
    return f"{units:.2f} {meta['unitLabel']}"


def save_to_db(res: StrategyResult, symbol: str, user: str, super_t: bool, top: StrategyResult | None):
    db = SessionLocal()
    db.add(
        TradeResult(
            userId=user,
            symbol=symbol,
            strategy=res.strategy,
            signal=res.signal,
            bias=res.bias,
            pattern=res.pattern,
            entry=res.entry,
            stopLoss=res.stopLoss,
            takeProfit=res.takeProfit,
            takeProfit2=res.takeProfit2,
            confidence=res.confidence,
            tradeType=res.tradeType,
            recommendedSize=res.recommendedSize,
            assetType=res.assetType,
            commentary=res.commentary,
            isSuperTrade=super_t,
            isTopPick=top is not None and res.strategy == top.strategy,
        )
    )
    db.commit()
    db.close()
def generate_prompt(strategy: str) -> str:
    return (
        f"You are an AI trading assistant. Provide a trade idea for {strategy} strategy as JSON with keys: "
        "signal, bias, pattern, entry, stopLoss, takeProfit, takeProfit2, confidence, tradeType, commentary."
    )


def check_confluence(group: list[StrategyResult]) -> bool:
    if len(group) < 3:
        return False
    avg_conf = sum(r.confidence for r in group) / len(group)
    same_bias = all(r.bias == group[0].bias for r in group)
    rr_ok = all(((r.takeProfit - r.entry) / abs(r.entry - r.stopLoss)) >= 1.5 for r in group)
    return avg_conf >= 78 and same_bias and rr_ok


@app.post("/analyze", response_model=FullAnalysis)
async def analyze_chart(
    file: UploadFile = File(...),
    portfolioSize: float = Form(10000),
    riskTolerance: str = Form("moderate"),
    userId: str = Form(...),
):
    image = Image.open(io.BytesIO(await file.read()))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    sym_meta = detect_symbol_and_metadata(img_b64)
    symbol_fallback = re.sub(r"[\d\-]+", "", sym_meta.get("symbol", "")).upper()
    meta = INSTRUMENT_MAP.get(
        symbol_fallback,
        {
            "assetType": sym_meta.get("assetType", "Unknown"),
            "pointValue": sym_meta.get("pointValue", 1),
            "tickSize": sym_meta.get("tickSize", 0.1),
            "unitLabel": sym_meta.get("unitLabel", "units"),
        },
    )

    strategies = [
        "SMC",
        "Breakout",
        "Fibonacci",
        "PriceAction",
        "Reversal",
        "Trendline",
        "LiquiditySweep",
        "SupportResistance",
        "Scalping",
        "SupplyDemand",
        "NexusPulse",
    ]

    results: list[StrategyResult] = []

    for strat in strategies:
        try:
            prompt = generate_prompt(strat)
            resp = openai.chat.completions.create(
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
                max_tokens=1000,
            )
            match = re.search(r"{.*}", resp.choices[0].message.content, re.DOTALL)
            if not match:
                continue
            data = json.loads(match.group())
            if not {"signal", "entry", "stopLoss", "takeProfit"}.issubset(data):
                continue
            data["strategy"] = strat
            data["assetType"] = meta["assetType"]
            data["recommendedSize"] = calculate_recommended_size(
                data["entry"],
                data["stopLoss"],
                portfolioSize,
                riskTolerance,
                meta,
            )
            try:
                res = StrategyResult(**data)
                results.append(res)
            except ValidationError:
                continue
        except Exception:
            continue

    buy_group = [r for r in results if r.signal.lower() == "buy"]
    sell_group = [r for r in results if r.signal.lower() == "sell"]
    super_trade = check_confluence(buy_group) or check_confluence(sell_group)
    top_pick = max(results, key=lambda r: r.confidence) if results else None
    conflict = None
    if top_pick and any(r.signal != top_pick.signal for r in results):
        conflict = "⚠️ Conflicting trade signals detected. Proceed with caution."

    for r in results:
        save_to_db(r, symbol_fallback or sym_meta.get("symbol", ""), userId, super_trade, top_pick)

    return FullAnalysis(results=results, superTrade=super_trade, topPick=top_pick, conflictCommentary=conflict)

@app.get("/history/{user_id}")
def get_history(user_id: str):
    db = SessionLocal()
    trades = (
        db.query(TradeResult)
        .filter(TradeResult.userId == user_id)
        .order_by(TradeResult.timestamp.desc())
        .all()
    )
    db.close()

    def model_to_dict(obj):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

    return [model_to_dict(t) for t in trades]


@app.post("/analyze", response_model=FullAnalysis)
async def analyze_chart(
    file: UploadFile = File(...),
    portfolioSize: float = Form(10000),
    riskTolerance: str = Form("moderate"),
    userId: str = Form(...)
    tradeStyle: str = Form("Intraday"),
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
            # Optional takeProfit2
            json_data["takeProfit2"] = json_data.get("takeProfit2")
            result = StrategyResult(**json_data)
            results.append(result)
        except Exception as e:
            print(f"Error in {strategy}: {e}")
            continue

    super_trade = False
    top_pick = None
    conflict_commentary = None

    if len(results) >= 3:
        buy_signals = [r for r in results if r.signal.lower() == "buy"]
        sell_signals = [r for r in results if r.signal.lower() == "sell"]

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

    if top_pick and any(r.signal != top_pick.signal for r in results):
        conflict_commentary = "⚠️ Conflicting trade signals detected. Strategies are not fully aligned. Proceed with caution."

    for r in results:
        save_to_db(r, fallback_symbol or symbol_meta.get("symbol", ""), userId, super_trade, top_pick)

    return FullAnalysis(results=results, superTrade=super_trade, topPick=top_pick, conflictCommentary=conflict_commentary)
    
   def generate_prompt(strategy: str, tradeStyle: str = "Intraday") -> str:
    strategy_prompts = {
        "SMC": f"You are a professional trading assistant specialized in Smart Money Concepts (SMC). The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Identify market structure (HH, LL), BOS, CHoCH, valid OBs, liquidity sweeps, and FVGs.\n2. Only generate a trade if BOS is followed by a valid OB and liquidity sweep.\n3. Include TP1 based on nearby imbalance resolution, and TP2 based on extended structure targets if logical.\n4. Clearly explain why TP2 is justified or omitted.",
        
        "Breakout": f"You are a breakout strategy analyst. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Identify consolidation zones (tight range, 3+ candles).\n2. Detect breakout direction with volume/momentum and valid retest or continuation.\n3. TP1 should be the measured move or breakout projection; TP2 if a clean extended move is likely.\n4. Explain trade logic with respect to structure and projection.",
        
        "Fibonacci": f"You are a trading assistant specialized in Fibonacci-based analysis. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Detect swing high/low, draw retracement zones (0.618, 0.5, 0.382).\n2. Confirm reaction with candle patterns and trend.\n3. TP1 = prior high/low or structure level; TP2 = 1.272 or 1.618 extension.\n4. If only TP1 is safe, leave TP2 null.",
        
        "PriceAction": f"You are a trading assistant focused on price action setups. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Detect candlestick patterns (engulfing, pin bar, inside bar) at key S/R or structure levels.\n2. Confirm with confluence or trend.\n3. TP1 = first opposing structure; TP2 = next major level or range extreme if price momentum supports it.",
        
        "Reversal": f"You are a reversal expert. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Detect extended trend followed by reversal pattern (double top/bottom, divergence, H&S).\n2. TP1 = minor reversal target; TP2 = full counter-trend move if reversal is strong.\n3. Explain any weaknesses if TP2 is uncertain.",
        
        "Trendline": f"You are a trendline analysis specialist. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Identify 2+ touches trendline.\n2. Detect bounce or breakout with confirmation.\n3. TP1 = prior minor high/low or measured move; TP2 = next structural target.\n4. Skip TP2 if price has no space to move cleanly.",
        
        "LiquiditySweep": f"You specialize in liquidity grabs. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Find equal highs/lows, stop clusters.\n2. Confirm sweep + reversal structure.\n3. TP1 = post-sweep structural target; TP2 = extended run if impulsive move follows sweep.\n4. TP2 only if risk-reward remains favorable after TP1.",
        
        "SupportResistance": f"You are a trading assistant specialized in support and resistance trading strategies. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Identify recent key horizontal levels where price reacted at least twice — these include classic support/resistance zones and pivot levels (P, R1-R3, S1-S3).\n2. Calculate pivot points using recent price action: P = (High + Low + Close)/3. Then derive R1, R2, R3 and S1, S2, S3 accordingly.\n3. Analyze if price recently bounced from, rejected, or broke one of those levels.\n4. Confirm the setup using a strong candle signal or momentum (e.g., engulfing, pin bar, breakout candle).\n5. TP1 = nearest recent high/low or small range expansion. TP2 = full range expansion or next pivot level.\n6. Ensure logical SL and TP based on structure.\n\nIf no valid setup exists, return:\n{{\"superTrade\": false, \"commentary\": \"No valid trade near support, resistance, or pivot levels.\"}}",
        
        "Scalping": f"You are analyzing for high-frequency scalping setups on low timeframes (5m, 2m, 1m). The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Focus on micro price reactions at key levels (e.g., order blocks, VWAP, EMAs, round numbers, previous highs/lows, and intraday pivot levels).\n2. Detect quick rejection patterns like small engulfing candles, pin bars, or inside bar breakouts — especially near structure or S/R.\n3. Look for volume/momentum bursts, fast wicks, or imbalance zones suggesting liquidity grabs.\n4. Confirm setup with market structure (e.g., BOS, CHoCH) or rapid price flip.\n5. Entry should offer a tight stop-loss (SL) and fast TP1 (e.g., 1.2–1.5 RR within 3–6 candles max). TP2 can be a larger structure expansion if momentum continues.\n6. Annotate if setup is best suited for 1m, 2m, or 5m timeframe based on candle formation density and structure.\n\nIf no suitable setup exists, return:\n{{\"superTrade\": false, \"commentary\": \"No fast-reacting scalping setup detected on current price structure.\"}}",
        
        "SupplyDemand": f"You are a supply/demand analyst. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Detect fresh rally-base-drop or drop-base-rally zones.\n2. Confirm price return with rejection.\n3. TP1 = origin of imbalance; TP2 = next demand/supply zone if clear continuation exists.\n4. Omit TP2 if structure is messy.",
        
        "NexusPulse": f"You are a general assistant offering guidance when no strategy fits cleanly. The user is currently looking for a {tradeStyle} setup.\n\nInstructions:\n1. Analyze trend, price structure, volatility, and S/R.\n2. Provide TP1 = safest conservative target.\n3. Suggest TP2 only if there's logic for further move.\n4. Be clear in commentary if confidence or targets are soft."
    }

    fallback = (
        "\n\nIf no valid setup is found with strict criteria, slightly relax the conditions "
        "and attempt to generate the best possible trade idea. Use a confidence score between 60 and 70 "
        "and clearly explain any uncertainty or weakness in the commentary."
        "\n\nIf the market structure allows, provide two take profits:\n"
        "- TP1 (conservative target)\n"
        "- TP2 (extended/projection target)\n"
        "If only one TP1 makes sense, set TP2 to null or omit it."
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
        "  \"takeProfit2\": float (optional),\n"
        "  \"confidence\": float (0 to 100),\n"
        "  \"tradeType\": \"Scalp, Intraday, or Swing\",\n"
        "  \"commentary\": \"Detailed explanation using logic, structure, risk-reward, confluence.\"\n"
        "}\n"
        "⚠️ Return only a single flat JSON object. Do not return any text before or after."
    )

    return strategy_prompts.get(strategy, "You are a trading assistant.") + fallback + schema
