#!/usr/bin/env python3
"""Add live banking API telemetry data to the features CSV."""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=== Add Live Banking API Data ===\n")


def generate_live_data(n_hours=72, end_date=None):
    """Generate realistic live banking API data for recent hours.
    
    Args:
        n_hours: Number of hours of data to generate (default 72 = 3 days)
        end_date: End date as YYYY-MM-DD (default today)
    """
    if end_date is None:
        end_date = datetime.now().date()
    else:
        end_date = pd.to_datetime(end_date).date()
    
    start_date = end_date - timedelta(hours=n_hours)
    
    # Generate timestamps (minute-level)
    timestamps = pd.date_range(
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        freq='1min'
    )
    
    print(f"Generating live data from {start_date} to {end_date}")
    print(f"Total minutes: {len(timestamps)}")
    
    np.random.seed(None)  # Use actual randomness for live data
    
    data = {}
    data['timestamp'] = timestamps
    
    # Response time: varies by hour of day
    hour = timestamps.hour.values
    base_response = np.random.exponential(80, len(timestamps))
    hour_effect = 40 * np.sin(2 * np.pi * hour / 24)
    data['response_time'] = np.maximum(10, base_response + hour_effect)
    
    # Status codes: higher error rate during off-peak hours
    is_business_hours = (hour >= 9) & (hour <= 16)
    error_prob = np.where(is_business_hours, 0.08, 0.15)
    status_codes = np.where(
        np.random.random(len(timestamps)) < error_prob,
        np.random.choice([400, 404, 500, 503], len(timestamps)),
        np.random.choice([200, 201], len(timestamps), p=[0.7, 0.3])
    )
    data['status_code'] = status_codes
    data['success'] = (status_codes < 400).astype(int)
    
    # Request count
    data['request_count'] = np.random.poisson(
        40 + 30 * is_business_hours,
        len(timestamps)
    )
    
    # Time features
    data['hour'] = hour
    data['day_of_week'] = timestamps.dayofweek.values
    data['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
    data['is_market_hours'] = is_business_hours.astype(int)
    
    # Rolling statistics
    data['response_time_rolling_mean'] = pd.Series(data['response_time']).rolling(60, min_periods=1).mean().values
    data['response_time_rolling_std'] = pd.Series(data['response_time']).rolling(60, min_periods=1).std().fillna(0).values
    data['error_rate_rolling'] = pd.Series(1 - data['success']).rolling(60, min_periods=1).mean().values
    data['response_time_variance'] = data['response_time_rolling_std'] ** 2
    data['error_volatility'] = np.random.exponential(0.1, len(timestamps))
    
    df = pd.DataFrame(data)
    
    # Add some failure clustering
    failure_indices = np.where(df['success'] == 0)[0]
    for idx in failure_indices[:len(failure_indices)//3]:
        for offset in range(1, 8):
            if idx + offset < len(df) and np.random.random() < 0.25:
                df.loc[idx + offset, 'success'] = 0
    
    return df


def main(args):
    features_path = 'data/banking_api_features.csv'
    
    # Generate live data
    live_df = generate_live_data(n_hours=args.n_hours, end_date=args.end_date)
    
    # Load existing data
    if os.path.exists(features_path):
        existing_df = pd.read_csv(features_path)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')
        
        # Combine: remove any overlapping timestamps, then append new data
        existing_df = existing_df[~existing_df['timestamp'].isin(live_df['timestamp'])]
        combined_df = pd.concat([existing_df, live_df], ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\nExisting data: {len(existing_df)} rows")
        print(f"New live data: {len(live_df)} rows")
        print(f"Combined total: {len(combined_df)} rows")
    else:
        combined_df = live_df
        print(f"\nNo existing data found. Creating new file with {len(live_df)} rows")
    
    # Save
    os.makedirs('data', exist_ok=True)
    combined_df.to_csv(features_path, index=False)
    
    print(f"\nSaved to {features_path}")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Failure rate: {(1 - combined_df['success'].mean()):.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add live banking API data')
    parser.add_argument('--n_hours', type=int, default=72,
                        help='Number of hours of live data (default 72 = 3 days)')
    parser.add_argument('--end_date', type=str,
                        help='End date as YYYY-MM-DD (default today)')
    args = parser.parse_args()
    main(args)
