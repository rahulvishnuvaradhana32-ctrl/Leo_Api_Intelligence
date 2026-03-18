#!/usr/bin/env python3
"""
Script to run all simulated APIs simultaneously.
Each API runs on a different port.
"""

import subprocess
import sys
import time
import os

# API configurations
APIS = [
    {"name": "weather", "file": "apis.weather_api", "port": 8001},
    {"name": "payment", "file": "apis.payment_api", "port": 8002},
    {"name": "user_auth", "file": "apis.user_auth_api", "port": 8003},
    {"name": "database", "file": "apis.database_api", "port": 8004},
    {"name": "messaging", "file": "apis.messaging_api", "port": 8005},
]

def run_api(api_config):
    """Run a single API using uvicorn."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{api_config['file'].replace('src/apis/', '').replace('.py', '')}:app",
        "--host", "127.0.0.1",
        "--port", str(api_config['port']),
        "--reload"  # Enable reload for development
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(os.getcwd(), 'src')

    print(f"Starting {api_config['name']} API on port {api_config['port']}")
    return subprocess.Popen(cmd, cwd=os.getcwd(), env=env)

def main():
    """Start all APIs."""
    processes = []

    try:
        for api in APIS:
            proc = run_api(api)
            processes.append(proc)
            time.sleep(1)  # Brief delay between starts

        print("\nAll APIs started successfully!")
        print("Press Ctrl+C to stop all APIs")

        # Wait for all processes
        for proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\nStopping all APIs...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        print("All APIs stopped.")

if __name__ == "__main__":
    main()