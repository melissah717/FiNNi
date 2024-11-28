from utils import load
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
import logging
import time

class DataPipeline:
    def __init__(self, date, rolling_window, trend_target_time, prediction_mode=False):
        self.date = date
        self.rolling_window = rolling_window
        self.trend_target_time = trend_target_time
        self.filtered_df = None
        self.prediction_mode = prediction_mode
        self.last_used_index = -1
        
    def load_data(self):
        print(f"Loading {self.date}")
        self.filtered_df = load(self.date)
        # if self.filtered_df.empty or len(self.filtered_df) < self.rolling_window:
        #     raise ValueError("No data available")
        print(self.filtered_df)

        # Filter data for the current day only to avoid using data from previous days
        self.filtered_df['timestamp'] = pd.to_datetime(self.filtered_df['timestamp']).dt.round('min')
        self.filtered_df = self.filtered_df[self.filtered_df['timestamp'].dt.date == pd.to_datetime(self.date).date()]

        target_time = pd.to_datetime(f"{self.date} {self.trend_target_time}")
        self.filtered_df = self.filtered_df[self.filtered_df['timestamp'] <= target_time]

        return self

    def process_features(self):
        print("Processing features...")
        df = self.filtered_df

        # Proceed with available data in prediction mode even if rolling window is not complete
        if len(df) < self.rolling_window:
            if self.prediction_mode:
                missing_points = self.rolling_window - len(df)
                logging.warning(f"Not enough data for a full rolling window. Missing {missing_points} data points. Proceeding with available data in prediction mode.")
            else:
                missing_points = self.rolling_window - len(df)
                print(f"Waiting for more data... {missing_points} data points left until first rolling window calculation can be performed.")
                return self

        # Processing features
        df['total_volume'] = df['options_data'].apply(lambda x: sum(opt['Volume'] for opt in x))
        df['pcr_mean'] = df['put_call_ratio'].rolling(self.rolling_window, min_periods=1).mean()
        df['pcr_std'] = df['put_call_ratio'].rolling(self.rolling_window, min_periods=1).std()
        df['pcr_zscore'] = (df['put_call_ratio'] - df['pcr_mean']) / df['pcr_std']
        df['volume_mean'] = df['total_volume'].rolling(self.rolling_window, min_periods=1).mean()
        df['volume_std'] = df['total_volume'].rolling(self.rolling_window, min_periods=1).std()
        df['volume_zscore'] = (df['total_volume'] - df['volume_mean']) / df['volume_std']
        df['volume_ma_short'] = df['total_volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ma_long'] = df['total_volume'].rolling(window=60, min_periods=1).mean()

        df = df.dropna()
        self.filtered_df = df
        print(f"Filtered DataFrame length after processing features: {len(self.filtered_df)}")
        return self

    
    def validate(self):
        print("Validating data...")
        required_data_points = 10
        if len(self.filtered_df) < required_data_points:
            raise ValueError(f"Not enough data available. Requires at least {required_data_points} rows.")
        return self
        
    def get_features_targets(self):
        print("Generating features and targets...")
        df = self.filtered_df.copy()

        target_time = pd.to_datetime(f"{self.date} {self.trend_target_time}")
        closest_target_row = df.iloc[(df['timestamp'] - target_time).abs().argsort()[:1]]


        if closest_target_row.empty:
            raise ValueError(f"No available data close to target time {target_time}")

        y_value = closest_target_row['underlying_price'].values[0]

        # Add underlying_price as a feature along with existing features
        features = df[['pcr_zscore', 'volume_zscore', 'volume_ma_short', 'volume_ma_long', 'underlying_price']].values
        targets = [y_value] * len(features)

        X = pd.DataFrame(features, columns=['pcr_zscore', 'volume_zscore', 'volume_ma_short', 'volume_ma_long', 'underlying_price'])
        y = pd.Series(targets)

        print(f"Generated features: {X.shape}, targets: {y.shape}")

        if X.empty or y.empty:
            raise ValueError("Generated features or targets are empty. Please check the data processing.")

        return X.values, y.values.reshape(-1, 1)
    
    def get_features_for_minute(self, timestamp):
        """
        Get features up to the given minute for real-time prediction.
        """
        filtered_df = self.filtered_df[self.filtered_df['timestamp'] <= timestamp]
        if filtered_df.empty:
            print(f"No available data before {timestamp}.")
            raise ValueError(f"No available data before {timestamp}")

        latest_row = filtered_df.iloc[-1]

        # Extract the relevant features from the latest row
        features = latest_row[['pcr_zscore', 'volume_zscore', 'volume_ma_short', 'volume_ma_long', 'underlying_price']].values
        return features.reshape(1, -1)
    

    def get_next_row(self, rolling_window):
        """
        Get the next row of data that is closest to the current time.
        Takes into account the rolling window and 15-minute delay.
        """
        self.load_data()
        self.process_features()

        # Current time adjusted for the data delay
        current_time = pd.Timestamp(datetime.now(tz=pytz.UTC)).floor('min') - timedelta(minutes=15)
        print(f"Current adjusted time: {current_time}")

        if self.filtered_df['timestamp'].dt.tz is None:
            self.filtered_df['timestamp'] = self.filtered_df['timestamp'].dt.tz_localize('UTC')

        available_rows = self.filtered_df[self.filtered_df['timestamp'] <= current_time]
        print(available_rows)
        if available_rows.empty:
            print("No rows available up to the current adjusted time. Going back to sleep for one minute...")
            return None
        window_start_time = current_time - timedelta(minutes=self.rolling_window)
        rolling_data = self.filtered_df[(self.filtered_df['timestamp'] >= window_start_time) & 
                                        (self.filtered_df['timestamp'] <= current_time)]

        # Ensure that we have enough rolling window data
        if len(rolling_data) < self.rolling_window:
            print("Not enough rolling window to calculate features, going back to sleep for one minute....")
            return None

        # Select the row closest to the current adjusted time
        next_row = available_rows.iloc[-1]  # Get the most recent row up to the current time
        print(f"Retrieving features for timestamp: {next_row['timestamp']} (current adjusted time: {current_time})")
        return next_row