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

# --- ClarifyRequest model ---
class ClarifyRequest(BaseModel):
    trade: StrategyResult
    question: str


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
    print(f"✅ Trade saved for user: {user}, strategy: {res.strategy}")
    db.close()
def generate_prompt(strategy: str, tradeStyle: str = "Intraday") -> str:
    strategy_prompts = {
        "SMC": f"You are a professional trading assistant specialized in Smart Money Concepts (SMC). The user is currently looking for a {tradeStyle} setup. Tailor your structure analysis to the selected trade style:\n- For Scalping: Use 1m, 2m, 3m, 5m charts.\n- For Intraday: Use 5m, 15m, 30m, 1h charts.\n- For Swing: Use 1h, 4h, daily charts.\nAlways mention the specific timeframe(s) you used for structure or pattern detection in your commentary.\n\nInstructions:\n1. Identify market structure (HH, LL), BOS, CHoCH, valid OBs, liquidity sweeps, and FVGs.\n2. Only generate a trade if BOS is followed by a valid OB and liquidity sweep.\n3. Include TP1 based on nearby imbalance resolution, and TP2 based on extended structure targets if logical.\n4. Clearly explain why TP2 is justified or omitted.",
        "Breakout": f"You are a breakout strategy analyst. The user is currently looking for a {tradeStyle} setup. Focus your analysis on 5m, 15m, 30m, or 1h charts for intraday trades; use 1m-5m for scalping; use 1h, 4h, or daily for swing. Clearly state which timeframe structure you are analyzing in your commentary.\n\nInstructions:\n1. Identify consolidation zones (tight range, 3+ candles).\n2. Detect breakout direction with volume/momentum and valid retest or continuation.\n3. TP1 should be the measured move or breakout projection; TP2 if a clean extended move is likely.\n4. Explain trade logic with respect to structure and projection.",
        "Fibonacci": f"You are a trading assistant specialized in Fibonacci-based analysis. The user is currently looking for a {tradeStyle} setup. Use 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Indicate the timeframe where the swing or retracement is most visible in your commentary.\n\nInstructions:\n1. Detect swing high/low, draw retracement zones (0.618, 0.5, 0.382).\n2. Confirm reaction with candle patterns and trend.\n3. TP1 = prior high/low or structure level; TP2 = 1.272 or 1.618 extension.\n4. If only TP1 is safe, leave TP2 null.",
        "PriceAction": f"You are a trading assistant focused on price action setups. The user is currently looking for a {tradeStyle} setup. Use 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Clearly mention the timeframe where the candlestick pattern or key level is detected in your commentary.\n\nInstructions:\n1. Detect candlestick patterns (engulfing, pin bar, inside bar) at key S/R or structure levels.\n2. Confirm with confluence or trend.\n3. TP1 = first opposing structure; TP2 = next major level or range extreme if price momentum supports it.",
        "Reversal": f"You are a reversal expert. The user is currently looking for a {tradeStyle} setup. Focus your reversal analysis on 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Indicate the timeframe where the reversal pattern is most clearly visible in your commentary.\n\nInstructions:\n1. Detect extended trend followed by reversal pattern (double top/bottom, divergence, H&S).\n2. TP1 = minor reversal target; TP2 = full counter-trend move if reversal is strong.\n3. Explain any weaknesses if TP2 is uncertain.",
        "Trendline": f"You are a trendline analysis specialist. The user is currently looking for a {tradeStyle} setup. Use 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Explicitly state the timeframe where the trendline is drawn or broken in your commentary.\n\nInstructions:\n1. Identify 2+ touches trendline.\n2. Detect bounce or breakout with confirmation.\n3. TP1 = prior minor high/low or measured move; TP2 = next structural target.\n4. Skip TP2 if price has no space to move cleanly.",
        "LiquiditySweep": f"You specialize in liquidity grabs. The user is currently looking for a {tradeStyle} setup. Focus your sweep detection on 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Clearly mention the timeframe where the sweep and reversal occur in your commentary.\n\nInstructions:\n1. Find equal highs/lows, stop clusters.\n2. Confirm sweep + reversal structure.\n3. TP1 = post-sweep structural target; TP2 = extended run if impulsive move follows sweep.\n4. TP2 only if risk-reward remains favorable after TP1.",
        "SupportResistance": f"You are a trading assistant specialized in support and resistance trading strategies. The user is currently looking for a {tradeStyle} setup. Focus your analysis on 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Always indicate the timeframe where the key level or pivot is most relevant in your commentary.\n\nInstructions:\n1. Identify recent key horizontal levels where price reacted at least twice — these include classic support/resistance zones and pivot levels (P, R1-R3, S1-S3).\n2. Calculate pivot points using recent price action: P = (High + Low + Close)/3. Then derive R1, R2, R3 and S1, S2, S3 accordingly.\n3. Analyze if price recently bounced from, rejected, or broke one of those levels.\n4. Confirm the setup using a strong candle signal or momentum (e.g., engulfing, pin bar, breakout candle).\n5. TP1 = nearest recent high/low or small range expansion. TP2 = full range expansion or next pivot level.\n6. Ensure logical SL and TP based on structure.\n\nIf no valid setup exists, return:\n{{\"superTrade\": false, \"commentary\": \"No valid trade near support, resistance, or pivot levels.\"}}",
        "Scalping": f"You are analyzing for high-frequency scalping setups. The user is currently looking for a {tradeStyle} setup.\nUse the 1m, 2m, 3m, and 5m charts as your reference timeframes. Indicate the specific timeframe where the setup is visible in your commentary.\n\nInstructions:\n1. Focus on micro price reactions at key levels (e.g., order blocks, VWAP, EMAs, round numbers, previous highs/lows, and intraday pivot levels).\n2. Detect quick rejection patterns like small engulfing candles, pin bars, or inside bar breakouts — especially near structure or S/R.\n3. Look for volume/momentum bursts, fast wicks, or imbalance zones suggesting liquidity grabs.\n4. Confirm setup with market structure (e.g., BOS, CHoCH) or rapid price flip.\n5. Entry should offer a tight stop-loss (SL) and fast TP1 (e.g., 1.2–1.5 RR within 3–6 candles max). TP2 can be a larger structure expansion if momentum continues.\n6. Annotate if setup is best suited for 1m, 2m, 3m, or 5m timeframe based on candle formation density and structure.\n\nIf no suitable setup exists, return:\n{{\"superTrade\": false, \"commentary\": \"No fast-reacting scalping setup detected on current price structure.\"}}",
        "SupplyDemand": f"You are a supply/demand analyst. The user is currently looking for a {tradeStyle} setup. Use 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Always specify the timeframe where the supply/demand zone is detected in your commentary.\n\nInstructions:\n1. Detect fresh rally-base-drop or drop-base-rally zones.\n2. Confirm price return with rejection.\n3. TP1 = origin of imbalance; TP2 = next demand/supply zone if clear continuation exists.\n4. Omit TP2 if structure is messy.",
        "NexusPulse": f"You are a general assistant offering guidance when no strategy fits cleanly. The user is currently looking for a {tradeStyle} setup. Use 5m, 15m, 30m, or 1h charts for intraday; 1m-5m for scalping; 1h, 4h, or daily for swing. Always indicate the timeframe where the pattern is most clearly visible in your commentary.\n\nInstructions:\n1. Analyze trend, price structure, volatility, and S/R.\n2. Provide TP1 = safest conservative target.\n3. Suggest TP2 only if there's logic for further move.\n4. Be clear in commentary if confidence or targets are soft."
    }

    fallback = (
        "\n\nIf no valid setup is found with strict criteria, still respond with a full JSON object using 'N/A', 0, or null values where necessary."
        "\n\nExample fallback values:\n"
        "\"signal\": \"N/A\",\n"
        "\"bias\": \"N/A\",\n"
        "\"pattern\": \"No valid setup detected\",\n"
        "\"entry\": 0,\n"
        "\"stopLoss\": 0,\n"
        "\"takeProfit\": 0,\n"
        "\"takeProfit2\": null,\n"
        "\"confidence\": 0,\n"
        "\"tradeType\": \"N/A\",\n"
        "\"commentary\": \"No setup found meeting the criteria.\"\n"
        "\nReturn all fields exactly as shown in the schema even if no trade is detected."
    )

    schema = (
        "\n\nRespond only with a JSON object using this exact schema:\n"
        "{\n"
        f"  \"strategy\": \"{strategy}\",\n"
        "  \"signal\": \"Buy or Sell or N/A\",\n"
        "  \"bias\": \"Bullish or Bearish or N/A\",\n"
        "  \"pattern\": \"Describe the key pattern you used or say 'None'\",\n"
        "  \"entry\": float,\n"
        "  \"stopLoss\": float,\n"
        "  \"takeProfit\": float,\n"
        "  \"takeProfit2\": float (optional or null),\n"
        "  \"confidence\": float (0 to 100),\n"
        "  \"tradeType\": \"Scalp, Intraday, Swing, or N/A\",\n"
        "  \"commentary\": \"Detailed explanation or reason for fallback.\"\n"
        "}\n"
        "⚠️ Return only a single flat JSON object. Do not return any text before or after."
    )

    prompt_intro = strategy_prompts.get(strategy, f"You are a trading assistant. The user is currently looking for a {tradeStyle} setup.")
    return prompt_intro + fallback + schema


def check_confluence(group: list[StrategyResult]) -> bool:
    if len(group) < 2:
        return False
    same_bias = all(r.bias == group[0].bias for r in group)
    same_signal = all(r.signal == group[0].signal for r in group)
    entry_range_ok = max(r.entry for r in group) - min(r.entry for r in group) <= 0.015 * sum(r.entry for r in group) / len(group)
    return same_bias and same_signal and entry_range_ok



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
    userId: str = Form(...),
    tradeStyle: str = Form("Intraday"),
    assetType: str = Form("Forex"),
    language: str = Form("en"),
):
    # Log incoming form field values
    print("Received request with:")
    print("Portfolio Size:", portfolioSize)
    print("Risk:", riskTolerance)
    print("Asset Type:", assetType)
    print("Language:", language)
    print("Chart Timeframe:", tradeStyle)

    image = Image.open(io.BytesIO(await file.read()))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    print("Image received and decoded successfully.")

    symbol_meta = detect_symbol_and_metadata(img_b64)
    fallback_symbol = re.sub(r'[\d\-]+', '', symbol_meta.get("symbol", "")).upper()
    meta = INSTRUMENT_MAP.get(fallback_symbol, {
        "assetType": symbol_meta.get("assetType", "Unknown"),
        "pointValue": symbol_meta.get("pointValue", 1),
        "tickSize": symbol_meta.get("tickSize", 0.1),
        "unitLabel": symbol_meta.get("unitLabel", "units")
    })
    if assetType != "Unknown":
        meta["assetType"] = assetType

    strategies = ["SMC", "Breakout", "Fibonacci", "PriceAction", "Reversal",
                  "Trendline", "LiquiditySweep", "SupportResistance", "Scalping", "SupplyDemand", "NexusPulse"]
    results = []

    for strategy in strategies:
        try:
            context_intro = (
                f"Respond only in {language.upper()}. The user is analyzing a {assetType} chart using a {tradeStyle} strategy. "
                f"Tailor the analysis to match the risk level: {riskTolerance}. The portfolio is approximately ${portfolioSize}.\n\n"
            )
            prompt = context_intro + generate_prompt(strategy, tradeStyle)
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
            json_data["commentary"] = f"(Structure observed on {tradeStyle} timeframe.) " + json_data["commentary"]
            json_data["assetType"] = assetType or meta["assetType"]
            if "entry" in json_data and "stopLoss" in json_data:
                json_data["recommendedSize"] = calculate_recommended_size(
                    json_data["entry"], json_data["stopLoss"], portfolioSize, riskTolerance, meta
                )
            # Optional takeProfit2
            json_data["takeProfit2"] = json_data.get("takeProfit2")
            # Insert confidence/rr/tradeStyle logic after confidence is set
            if "confidence" in json_data and "takeProfit" in json_data and "entry" in json_data and "stopLoss" in json_data:
                rr = (json_data["takeProfit"] - json_data["entry"]) / max(0.01, abs(json_data["entry"] - json_data["stopLoss"]))
                if rr >= 2.0:
                    json_data["confidence"] += 5
                if tradeStyle == "Scalp" and strategy in ["Scalping", "LiquiditySweep", "Trendline"]:
                    json_data["confidence"] += 5
                json_data["confidence"] = min(100, max(50, json_data["confidence"]))
                if json_data["confidence"] > 75:
                    json_data["commentary"] += " ⚡ High-probability setup detected."
            # Only append if not a fallback/incomplete response
            if json_data.get("signal") != "N/A" and json_data.get("entry", 0) > 0 and json_data.get("takeProfit", 0) > 0:
                result = StrategyResult(**json_data)
                results.append(result)
            else:
                print(f"Filtered out fallback or incomplete strategy: {strategy}")
        except Exception as e:
            print(f"Error in {strategy}: {e}")
            continue

    super_trade = False
    top_pick = None
    conflict_commentary = None

    if len(results) >= 3:
        buy_signals = [r for r in results if r.signal.lower() == "buy"]
        sell_signals = [r for r in results if r.signal.lower() == "sell"]

        if check_confluence(buy_signals):
            super_trade = True
        elif check_confluence(sell_signals):
            super_trade = True

    if results:
        top_pick = max(results, key=lambda r: (r.confidence, (r.takeProfit - r.entry) / max(0.01, abs(r.entry - r.stopLoss))))

    if top_pick and any(r.signal != top_pick.signal for r in results):
        conflict_commentary = "⚠️ Conflicting trade signals detected. Strategies are not fully aligned. Proceed with caution."

    if top_pick:
        save_to_db(top_pick, fallback_symbol or symbol_meta.get("symbol", ""), userId, super_trade, top_pick)

    print("Returning response with", len(results), "strategy results.")
    return FullAnalysis(results=results, superTrade=super_trade, topPick=top_pick, conflictCommentary=conflict_commentary)


from fastapi import Body

# --- Clarify endpoint ---
@app.post("/clarify")
async def clarify_trade(data: ClarifyRequest):
    prompt = f"""
    You are a trading assistant helping clarify a trade setup.
    Here is the strategy output:

    {json.dumps(data.trade.dict(), indent=2)}

    The user asks: \"{data.question}\"

    Provide a helpful, expert-level explanation that references the trade logic, confidence, entry, and price structure.
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"response": response.choices[0].message.content}

# --- FeedbackRequest model ---
class FeedbackRequest(BaseModel):
    tradeId: int
    wasWin: bool


# --- TradeFeedback database model ---
class TradeFeedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    tradeId = Column(Integer, index=True)
    wasWin = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Ensure feedback table is created
Base.metadata.create_all(bind=engine)


# --- Feedback endpoint ---
@app.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest):
    db = SessionLocal()
    new_feedback = TradeFeedback(tradeId=feedback.tradeId, wasWin=feedback.wasWin)
    db.add(new_feedback)
    db.commit()
    db.close()
    return {"status": "success", "message": "Feedback recorded"}