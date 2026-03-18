from fastapi import FastAPI, HTTPException
import random
import time
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Crypto Price API", description="Mock cryptocurrency price service")

class CryptoRequest(BaseModel):
    symbol: str = "BTC"
    currency: str = "USD"

CRYPTO_PRICES = {
    "BTC": {"base_price": 45000.0, "volatility": 0.05},
    "ETH": {"base_price": 2500.0, "volatility": 0.06},
    "ADA": {"base_price": 0.5, "volatility": 0.08},
    "SOL": {"base_price": 100.0, "volatility": 0.07},
    "DOT": {"base_price": 8.0, "volatility": 0.09}
}

@app.get("/crypto_price")
async def get_crypto_price(symbol: str = "BTC", currency: str = "USD"):
    if symbol not in CRYPTO_PRICES:
        raise HTTPException(status_code=404, detail="Cryptocurrency not supported")

    # Simulate high-frequency trading delays
    base_delay = 0.02  # Very fast for crypto
    current_hour = datetime.now().hour

    # Crypto markets run 24/7, but higher volatility at certain hours
    # Peak volatility during Asian/European overlap (00:00-08:00 UTC)
    if 0 <= current_hour <= 8:
        base_delay *= 1.2

    # Add random network latency
    delay = random.uniform(base_delay * 0.5, base_delay * 2.0)
    time.sleep(delay)

    # Crypto-specific failure patterns
    # Exchange maintenance (rare but impactful)
    if random.random() < 0.003:
        raise HTTPException(status_code=503, detail="Exchange under maintenance")

    # High volatility circuit breakers
    if random.random() < 0.01:
        raise HTTPException(status_code=503, detail="Trading paused - extreme volatility")

    # API rate limits (common in crypto)
    if random.random() < 0.05:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Blockchain congestion
    if random.random() < 0.02:
        raise HTTPException(status_code=502, detail="Blockchain network congestion")

    # Simulate price with high volatility
    crypto = CRYPTO_PRICES[symbol]
    price_change = random.gauss(0, crypto["volatility"])
    current_price = crypto["base_price"] * (1 + price_change)

    # Ensure positive price
    current_price = max(current_price, 0.0001)

    return {
        "symbol": symbol,
        "currency": currency,
        "price": round(current_price, 2 if current_price > 1 else 6),
        "change_24h": round(price_change * 100, 2),
        "volume_24h": random.randint(1000000, 100000000),
        "market_cap": round(current_price * random.randint(10000000, 1000000000), 0),
        "timestamp": datetime.now().isoformat(),
        "exchange": random.choice(["Binance", "Coinbase", "Kraken", "Gemini"])
    }