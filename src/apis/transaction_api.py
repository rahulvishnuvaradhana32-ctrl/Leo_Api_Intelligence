from fastapi import FastAPI, HTTPException
import random
import time
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Transaction API", description="Mock banking transaction processing API")

class TransactionRequest(BaseModel):
    account_from: str
    account_to: str
    amount: float
    currency: str = "USD"
    transaction_type: str = "transfer"  # transfer, payment, deposit, withdrawal

@app.post("/process_transaction")
async def process_transaction(request: TransactionRequest):
    # Simulate transaction processing time
    base_delay = 0.8  # Banking transactions take longer
    current_hour = datetime.now().hour

    # Peak banking hours
    if 9 <= current_hour <= 11 or 13 <= current_hour <= 15:
        base_delay *= 1.5

    # Transaction complexity
    if request.transaction_type == "international":
        base_delay *= 2
    elif request.transaction_type == "crypto":
        base_delay *= 1.2

    # High-value transactions get extra scrutiny
    if request.amount > 10000:
        base_delay *= 1.5

    delay = random.uniform(base_delay * 0.8, base_delay * 1.2)
    time.sleep(delay)

    # Banking transaction failure patterns
    # Insufficient funds (8% chance)
    if random.random() < 0.08:
        raise HTTPException(status_code=402, detail="Insufficient funds")

    # Account frozen/suspended (3% chance)
    if random.random() < 0.03:
        raise HTTPException(status_code=423, detail="Account suspended")

    # Regulatory compliance check failure (2% chance)
    if random.random() < 0.02:
        raise HTTPException(status_code=403, detail="Transaction blocked - compliance")

    # System maintenance (1% chance)
    if random.random() < 0.01:
        raise HTTPException(status_code=503, detail="Banking system maintenance")

    # Fraud detection trigger (4% chance)
    if random.random() < 0.04:
        raise HTTPException(status_code=403, detail="Transaction flagged for review")

    # AML/KYC failure (1% chance)
    if request.amount > 5000 and random.random() < 0.02:
        raise HTTPException(status_code=403, detail="AML/KYC verification required")

    # Success
    transaction_id = f"txn_{random.randint(100000000, 999999999)}"
    fee = round(request.amount * 0.001, 2)  # 0.1% fee

    return {
        "status": "processed",
        "transaction_id": transaction_id,
        "amount": request.amount,
        "fee": fee,
        "net_amount": request.amount - fee,
        "currency": request.currency,
        "account_from": request.account_from,
        "account_to": request.account_to,
        "transaction_type": request.transaction_type,
        "processing_time": delay,
        "timestamp": datetime.now().isoformat(),
        "confirmation_code": f"CONF_{random.randint(10000, 99999)}"
    }

@app.get("/transaction_status/{transaction_id}")
async def get_transaction_status(transaction_id: str):
    # Check transaction status
    delay = random.uniform(0.1, 0.3)
    time.sleep(delay)

    # Status possibilities
    statuses = ["completed", "pending", "failed", "processing"]
    weights = [0.85, 0.10, 0.04, 0.01]  # Weighted random

    status = random.choices(statuses, weights=weights)[0]

    if status == "failed":
        reasons = [
            "Insufficient funds",
            "Account suspended",
            "Compliance block",
            "Technical error",
            "Timeout"
        ]
        reason = random.choice(reasons)
        return {
            "transaction_id": transaction_id,
            "status": status,
            "failure_reason": reason,
            "timestamp": datetime.now().isoformat()
        }

    return {
        "transaction_id": transaction_id,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }