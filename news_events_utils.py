import os
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
NEWS_BASE_URL = "https://newsdata.io/api/1"


def get_recent_news(symbol: str):
    url = f"{NEWS_BASE_URL}/news"
    params = {
        "apikey": NEWS_API_KEY,
        "q": symbol,
        "language": "en",
        "category": "business"
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        articles = res.json().get("results", [])
        return articles[:5]
    except Exception as e:
        print(f"❌ NewsData.io fetch failed: {e}")
        return []


def scrape_forex_factory():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        res = requests.get("https://www.forexfactory.com/calendar", headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        events = []
        for row in soup.select("tr.calendar__row"):
            time_el = row.select_one("td.calendar__time")
            currency_el = row.select_one("td.calendar__currency")
            impact_el = row.select_one("td.calendar__impact img")
            event_el = row.select_one("td.calendar__event")

            if time_el and currency_el and event_el:
                events.append({
                    "time": time_el.get_text(strip=True),
                    "currency": currency_el.get_text(strip=True),
                    "impact": impact_el.get("title", "Low") if impact_el else "Low",
                    "event": event_el.get_text(strip=True)
                })

        return events[:5]
    except Exception as e:
        print(f"❌ Forex Factory scrape failed: {e}")
        return []
