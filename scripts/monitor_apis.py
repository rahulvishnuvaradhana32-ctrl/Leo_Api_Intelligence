#!/usr/bin/env python3
"""
API Monitoring Script
Collects telemetry data from simulated APIs every minute.
Logs response time, error rate, request count, etc.
"""

import requests
import time
import csv
import os
from datetime import datetime
import random
import json

# API endpoints configuration
APIS = [
    {
        "name": "weather",
        "url": "http://127.0.0.1:8001/weather",
        "method": "GET",
        "params": {"city": "New York"}
    },
    {
        "name": "payment",
        "url": "http://127.0.0.1:8002/process_payment",
        "method": "POST",
        "json": {
            "amount": 99.99,
            "card_number": "4111111111111111",
            "expiry": "12/25"
        }
    },
    {
        "name": "user_auth",
        "url": "http://127.0.0.1:8003/login",
        "method": "POST",
        "json": {
            "username": "testuser",
            "password": "password123"
        }
    },
    {
        "name": "database",
        "url": "http://127.0.0.1:8004/query",
        "method": "POST",
        "json": {
            "query": "SELECT * FROM users LIMIT 10",
            "table": "users"
        }
    },
    {
        "name": "messaging",
        "url": "http://127.0.0.1:8005/send_message",
        "method": "POST",
        "json": {
            "to": "+1234567890",
            "message": "Test message",
            "priority": "normal"
        }
    }
]

DATA_DIR = "data"
TELEMETRY_FILE = os.path.join(DATA_DIR, "api_telemetry.csv")

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)

def initialize_csv():
    """Initialize CSV file with headers if it doesn't exist."""
    if not os.path.exists(TELEMETRY_FILE):
        with open(TELEMETRY_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'api_name', 'response_time', 'status_code',
                'success', 'error_type', 'request_count'
            ])

def call_api(api_config):
    """Call an API endpoint and measure response."""
    start_time = time.time()

    try:
        if api_config['method'] == 'GET':
            response = requests.get(
                api_config['url'],
                params=api_config.get('params', {}),
                timeout=10
            )
        elif api_config['method'] == 'POST':
            response = requests.post(
                api_config['url'],
                json=api_config.get('json', {}),
                timeout=10
            )

        response_time = time.time() - start_time
        status_code = response.status_code
        success = response.status_code < 400
        error_type = None if success else response.reason

    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        status_code = None
        success = False
        error_type = str(type(e).__name__)

    return {
        'response_time': response_time,
        'status_code': status_code,
        'success': success,
        'error_type': error_type
    }

def collect_telemetry():
    """Collect telemetry from all APIs."""
    timestamp = datetime.now().isoformat()
    telemetry_data = []

    # Simulate varying request load (1-5 requests per API per minute)
    request_count = random.randint(1, 5)

    for api in APIS:
        for _ in range(request_count):
            metrics = call_api(api)
            telemetry_data.append({
                'timestamp': timestamp,
                'api_name': api['name'],
                'response_time': metrics['response_time'],
                'status_code': metrics['status_code'],
                'success': metrics['success'],
                'error_type': metrics['error_type'],
                'request_count': 1  # Per call
            })

    return telemetry_data

def log_telemetry(telemetry_data):
    """Log telemetry data to CSV."""
    with open(TELEMETRY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        for data in telemetry_data:
            writer.writerow([
                data['timestamp'],
                data['api_name'],
                data['response_time'],
                data['status_code'],
                data['success'],
                data['error_type'],
                data['request_count']
            ])

def main():
    """Main monitoring loop."""
    ensure_data_dir()
    initialize_csv()

    print("Starting API monitoring for 5 minutes...")

    start_time = time.time()
    duration = 5 * 60  # 5 minutes

    while time.time() - start_time < duration:
        telemetry = collect_telemetry()
        log_telemetry(telemetry)

        # Log summary
        total_requests = len(telemetry)
        successful_requests = sum(1 for t in telemetry if t['success'])
        print(f"{datetime.now()}: {successful_requests}/{total_requests} successful requests")

        # Wait for 10 seconds
        time.sleep(10)

    print("Monitoring completed. Data saved to:", TELEMETRY_FILE)

if __name__ == "__main__":
    main()