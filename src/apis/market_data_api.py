from fastapi import FastAPI, HTTPException
import random
import time
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Market Data API", description="Mock comprehensive market data service")

class MarketDataRequest(BaseModel):
    symbols: list = ["AAPL", "GOOGL", "MSFT"]
    data_type: str = "quotes"  # quotes, news, fundamentals

@app.post("/market_data")
async def get_market_data(request: MarketDataRequest):
    # Simulate data retrieval time
    base_delay = 0.15
    current_hour = datetime.now().hour

    # Market data load during trading hours
    if 9 <= current_hour <= 16:
        base_delay *= 2

    # Data complexity factor
    if request.data_type == "fundamentals":
        base_delay *= 3
    elif request.data_type == "news":
        base_delay *= 1.5

    # Multiple symbols increase load
    base_delay *= (1 + len(request.symbols) * 0.1)

    delay = random.uniform(base_delay * 0.9, base_delay * 1.1)
    time.sleep(delay)

    # Market data failure patterns
    # Data feed outages (rare but critical)
    if random.random() < 0.005:
        raise HTTPException(status_code=503, detail="Data feed unavailable")

    # Regulatory circuit breakers
    if random.random() < 0.003:
        raise HTTPException(status_code=503, detail="Market-wide trading halt")

    # Data provider API limits
    if random.random() < 0.02:
        raise HTTPException(status_code=429, detail="API rate limit exceeded")

    # Network partitioning
    if random.random() < 0.01:
        raise HTTPException(status_code=502, detail="Data provider network error")

    # Invalid symbols
    if any(symbol not in ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"] for symbol in request.symbols):
        raise HTTPException(status_code=400, detail="Invalid symbol in request")

    # Simulate market data response
    data = {}
    for symbol in request.symbols:
        if request.data_type == "quotes":
            data[symbol] = {
                "price": round(random.uniform(100, 500), 2),
                "change": round(random.uniform(-10, 10), 2),
                "volume": random.randint(1000000, 50000000),
                "bid": round(random.uniform(95, 495), 2),
                "ask": round(random.uniform(105, 505), 2)
            }
        elif request.data_type == "news":
            data[symbol] = {
                "headlines": [f"Market Update for {symbol}" for _ in range(random.randint(1, 5))],
                "sentiment": random.choice(["positive", "negative", "neutral"])
            }
        elif request.data_type == "fundamentals":
            data[symbol] = {
                "pe_ratio": round(random.uniform(10, 50), 2),
                "market_cap": random.randint(100000000, 2000000000),
                "dividend_yield": round(random.uniform(0, 5), 2),
                "beta": round(random.uniform(0.5, 2.0), 2)
            }

    return {
        "data_type": request.data_type,
        "symbols": request.symbols,
        "data": data,
        "timestamp": datetime.now().isoformat(),
        "source": random.choice(["Bloomberg", "Reuters", "Refinitiv", "Morningstar"])
    }