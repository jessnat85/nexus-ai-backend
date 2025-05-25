# news_events_utils.py
import os
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Load API key from environment
NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
NEWS_BASE_URL = "https://newsdata.io/api/1"

def get_recent_news(query="US"):
    """Fetch top 5 financial news articles using NewsData.io"""
    url = f"{NEWS_BASE_URL}/news"
    params = {
        "apikey": NEWS_API_KEY,
        "q": query,
        "language": "en",
        "category": "business"
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        return data.get("results", [])[:5]
    except Exception as e:
        print(f"❌ NewsData.io fetch failed: {e}")
        return []

def scrape_forex_factory():
    """Scrape top 5 economic events from Forex Factory"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/"
        }
        res = requests.get("https://www.forexfactory.com/calendar", headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        events = []
        for row in soup.select("tr.calendar__row")[:10]:  # Scrape first 10 rows for safety
            time = row.select_one("td.calendar__time")
            currency = row.select_one("td.calendar__currency")
            impact = row.select_one("td.calendar__impact img")
            event = row.select_one("td.calendar__event")

            if time and currency and event:
                events.append({
                    "time": time.get_text(strip=True),
                    "currency": currency.get_text(strip=True),
                    "impact": impact["title"] if impact else "Low",
                    "event": event.get_text(strip=True)
                })

        return events[:5]
    except Exception as e:
        print(f"❌ Forex Factory scrape failed: {e}")
        return []
