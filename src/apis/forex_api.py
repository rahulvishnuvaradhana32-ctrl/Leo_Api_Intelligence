from fastapi import FastAPI, HTTPException
import random
import time
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Forex Rate API", description="Mock forex trading API")

class ForexRequest(BaseModel):
    base_currency: str = "USD"
    quote_currency: str = "EUR"
    amount: float = 1000.0

CURRENCY_PAIRS = {
    "EUR/USD": {"rate": 1.08, "volatility": 0.005},
    "GBP/USD": {"rate": 1.27, "volatility": 0.007},
    "USD/JPY": {"rate": 150.0, "volatility": 0.008},
    "USD/CHF": {"rate": 0.92, "volatility": 0.004},
    "AUD/USD": {"rate": 0.67, "volatility": 0.006}
}

@app.post("/forex_rate")
async def get_forex_rate(request: ForexRequest):
    pair = f"{request.base_currency}/{request.quote_currency}"

    if pair not in CURRENCY_PAIRS:
        raise HTTPException(status_code=400, detail="Unsupported currency pair")

    # Simulate processing time
    base_delay = 0.1
    current_hour = datetime.now().hour

    # Forex market hours: higher activity during London/New York overlap (13:00-17:00 UTC)
    # Assuming server is in UTC, adjust for local time
    if 13 <= current_hour <= 17:
        base_delay *= 1.5

    # Weekend/market closure effects
    current_day = datetime.now().weekday()
    if current_day >= 5:  # Saturday/Sunday
        base_delay *= 3

    delay = random.uniform(base_delay * 0.9, base_delay * 1.1)
    time.sleep(delay)

    # Forex-specific failure patterns
    # Bank holidays or market closures
    if current_day >= 5 and random.random() < 0.1:
        raise HTTPException(status_code=503, detail="Market closed - bank holiday")

    # High volatility events (news, economic data releases)
    if random.random() < 0.008:
        raise HTTPException(status_code=503, detail="Trading suspended - high volatility")

    # Liquidity issues during off-hours
    if not (8 <= current_hour <= 18) and random.random() < 0.02:
        raise HTTPException(status_code=502, detail="Insufficient liquidity")

    # Simulate rate with random fluctuation
    currency_data = CURRENCY_PAIRS[pair]
    rate_change = random.gauss(0, currency_data["volatility"])
    current_rate = currency_data["rate"] * (1 + rate_change)

    converted_amount = request.amount * current_rate

    return {
        "base_currency": request.base_currency,
        "quote_currency": request.quote_currency,
        "rate": round(current_rate, 4),
        "converted_amount": round(converted_amount, 2),
        "spread": round(random.uniform(0.0001, 0.001), 5),
        "timestamp": datetime.now().isoformat(),
        "market": "FX" if current_day < 5 else "OTC"
    }