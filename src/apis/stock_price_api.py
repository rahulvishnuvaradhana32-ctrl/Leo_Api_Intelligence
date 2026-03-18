from fastapi import FastAPI, HTTPException
import random
import time
from datetime import datetime
import math

app = FastAPI(title="Stock Price API", description="Mock stock price service API")

# Simulate stock prices with realistic volatility
STOCKS = {
    "AAPL": {"base_price": 150.0, "volatility": 0.02},
    "GOOGL": {"base_price": 2800.0, "volatility": 0.025},
    "MSFT": {"base_price": 300.0, "volatility": 0.018},
    "TSLA": {"base_price": 200.0, "volatility": 0.04},
    "AMZN": {"base_price": 3200.0, "volatility": 0.03}
}

@app.get("/stock_price")
async def get_stock_price(symbol: str = "AAPL"):
    if symbol not in STOCKS:
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    # Simulate processing time with market hours effect
    current_hour = datetime.now().hour
    base_delay = 0.05

    # Market hours: higher latency during trading hours (9:30-16:00)
    if 9 <= current_hour <= 16:
        base_delay *= 2
        # Pre-market/opening bell spikes
        if current_hour == 9:
            base_delay *= 3

    # Add random variation
    delay = random.uniform(base_delay * 0.8, base_delay * 1.2)
    time.sleep(delay)

    # Market failure patterns
    # Circuit breakers during high volatility
    if random.random() < 0.005:
        raise HTTPException(status_code=503, detail="Trading halted - circuit breaker")

    # Network congestion during market open/close
    if (current_hour == 9 or current_hour == 16) and random.random() < 0.02:
        raise HTTPException(status_code=504, detail="Gateway timeout - market congestion")

    # Data feed interruptions
    if random.random() < 0.01:
        raise HTTPException(status_code=502, detail="Bad Gateway - data feed unavailable")

    # Simulate price with random walk
    stock = STOCKS[symbol]
    price_change = random.gauss(0, stock["volatility"])
    current_price = stock["base_price"] * (1 + price_change)

    # Ensure positive price
    current_price = max(current_price, 0.01)

    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "change_percent": round(price_change * 100, 2),
        "volume": random.randint(100000, 10000000),
        "timestamp": datetime.now().isoformat(),
        "exchange": "NYSE" if symbol != "TSLA" else "NASDAQ"
    }