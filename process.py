from load import load
import pandas as pd
import numpy as np


def dynamic_trend_window(filtered_df, market_close):
    """
    Compute dynamic trend window based on time remaining to market close.
    """
    filtered_df['remaining_minutes'] = (market_close - filtered_df['timestamp']).dt.total_seconds() / 60
    filtered_df['dynamic_trend_window'] = filtered_df['remaining_minutes'] // 10
    return filtered_df


def process_and_save_data(date, skew_threshold, rolling_window, trend_window):
    """
    Process data: compute z-scores, features, and targets.
    """
    df = load(date)
    market_close = pd.Timestamp(f"{date} 16:00:00")
    
    if df.empty or len(df) < 30:
        print(f"Skipping {date} due to insufficient data")
        return None, None, None, None

    df['total_volume'] = df['options_data'].apply(lambda x: sum(opt['Volume'] for opt in x))

    df['pcr_mean'] = df['put_call_ratio'].rolling(rolling_window).mean()
    df['pcr_std'] = df['put_call_ratio'].rolling(rolling_window).std()
    df['pcr_zscore'] = (df['put_call_ratio'] - df['pcr_mean']) / df['pcr_std']

    # Compute volume Z-score and spikes
    df['volume_mean'] = df['total_volume'].rolling(rolling_window).mean()
    df['volume_std'] = df['total_volume'].rolling(rolling_window).std()
    df['volume_zscore'] = (df['total_volume'] - df['volume_mean']) / df['volume_std']

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df = dynamic_trend_window(df, market_close)

    # Drop NaNs
    df = df.dropna()

    if len(df) < trend_window:
        print(f"Insufficient data for {date}")
        return None, None, None, None

    # Define features and targets
    X = df[['pcr_zscore', 'volume_zscore', 'hour_sin', 'hour_cos']].iloc[:-trend_window].values
    y = df['underlying_price'].iloc[trend_window:].values

    if len(X) != len(y):
        print(f"X and y length mismatch for {date}")
        return None, None, None, None

    # Normalize targets
    y = y.reshape(-1, 1)
    return X, y, df