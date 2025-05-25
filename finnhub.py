# finnhub_utils.py
import os
import requests
from datetime import datetime, timedelta

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1"


def get_recent_news(symbol: str):
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)

    url = f"{BASE_URL}/company-news"
    params = {
        "symbol": symbol,
        "from": yesterday.isoformat(),
        "to": today.isoformat(),
        "token": FINNHUB_API_KEY
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()[:5]  # limit to top 5 news items
    except Exception as e:
        print(f"News fetch failed: {e}")
        return []


def get_economic_calendar():
    url = f"{BASE_URL}/calendar/economic"
    today = datetime.utcnow().date().isoformat()
    try:
        res = requests.get(url, params={"from": today, "to": today, "token": FINNHUB_API_KEY})
        res.raise_for_status()
        return res.json().get("economicCalendar", [])[:5]  # top 5 events
    except Exception as e:
        print(f"Calendar fetch failed: {e}")
        return []
