import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import openai
import base64
import logging

# Set your OpenAI API key here or via environment variable
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

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def ask_gpt_vision(prompt: str, image_bytes: bytes) -> str:
    try:
        base64_img = image_to_base64(image_bytes)
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial market AI that analyzes trading charts and returns the most likely strategy setup detected."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during GPT Vision analysis: {e}")
        return ""

@app.post("/analyze")
async def analyze_chart(file: UploadFile = File(...)):
    image_bytes = await file.read()

    strategies = [
        "Smart Money Concepts",
        "Breakout",
        "Fibonacci",
        "Price Action",
        "Reversal",
        "Trendline",
        "Liquidity Sweep",
        "Support/Resistance",
        "Scalping",
        "Order Block"
    ]

    parsed_results = []

    for strat in strategies:
        prompt = (
            f"Analyze this trading chart image using the {strat} strategy. "
            f"Return a single best signal found using this strategy. "
            "Respond strictly in this JSON format:\n"
            "{\n"
            f"  \"strategy\": \"{strat}\",\n"
            "  \"signal\": \"BUY or SELL\",\n"
            "  \"bias\": \"Bullish or Bearish\",\n"
            "  \"pattern\": \"Name of the pattern\",\n"
            "  \"entry\": float,\n"
            "  \"stopLoss\": float,\n"
            "  \"takeProfit\": float,\n"
            "  \"confidence\": float,\n"
            "  \"commentary\": \"Short explanation of the setup\"\n"
            "}"
        )

        raw_output = ask_gpt_vision(prompt, image_bytes)
        logging.info(f"Raw GPT output ({strat}):\n{raw_output}")

        try:
            import json
            data = json.loads(raw_output)
            validated = StrategyResult(**data)
            parsed_results.append(validated)
        except Exception as e:
            logging.error(f"Error in {strat} analysis: {e}")

    if not parsed_results:
        return {"error": "No valid strategy analysis detected."}

    best = max(parsed_results, key=lambda x: x.confidence)
    return best
