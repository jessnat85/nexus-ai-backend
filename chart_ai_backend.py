import os
import io
import re
import json
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
            prompt = generate_prompt(strategy, tradeStyle)
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
            json_data["assetType"] = meta["assetType"]
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

    return FullAnalysis(results=results, superTrade=super_trade, topPick=top_pick, conflictCommentary=conflict_commentary)