from utils import load
import pandas as pd

class DataPipeline:
    def __init__(self, date, rolling_window, trend_target_time):
        self.date = date
        self.rolling_window = rolling_window
        self.trend_target_time = trend_target_time
        self.filtered_df = None
        
    def load_data(self):
        print(f"Loading {self.date}")
        self.filtered_df = load(self.date)
        if self.filtered_df.empty or len(self.filtered_df) < 30:
            raise ValueError("No data available")
        
        self.filtered_df['timestamp'] = pd.to_datetime(self.filtered_df['timestamp']).dt.round('min')
        target_time = pd.to_datetime(f"{self.date} {self.trend_target_time}")
        self.filtered_df = self.filtered_df[self.filtered_df['timestamp'] <= target_time]

        return self


    def process_features(self):
        print("Processing features...")
        df = self.filtered_df
        df['total_volume'] = df['options_data'].apply(lambda x: sum(opt['Volume'] for opt in x))
        df['pcr_mean'] = df['put_call_ratio'].rolling(self.rolling_window).mean()
        df['pcr_std'] = df['put_call_ratio'].rolling(self.rolling_window).std()
        df['pcr_zscore'] = (df['put_call_ratio'] - df['pcr_mean']) / df['pcr_std']
        df['volume_mean'] = df['total_volume'].rolling(self.rolling_window).mean()
        df['volume_std'] = df['total_volume'].rolling(self.rolling_window).std()
        df['volume_zscore'] = (df['total_volume'] - df['volume_mean']) / df['volume_std']
        df['volume_ma_short'] = df['total_volume'].rolling(window=5).mean()
        df['volume_ma_long'] = df['total_volume'].rolling(window=60).mean()
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

        features = df[['pcr_zscore', 'volume_zscore', 'volume_ma_short', 'volume_ma_long']].values
        targets = [y_value] * len(features) 

        X = pd.DataFrame(features, columns=['pcr_zscore', 'volume_zscore', 'volume_ma_short', 'volume_ma_long'])
        y = pd.Series(targets)

        print(f"Generated features: {X.shape}, targets: {y.shape}")

        if X.empty or y.empty:
            raise ValueError("Generated features or targets are empty. Please check the data processing.")

        return X.values, y.values.reshape(-1, 1)
    
    def get_features_for_minute(self, timestamp):
        """
        Get features up to the given minute for real-time prediction.
        """
        filtered_df = self.filtered_df[self.filtered_df['timestamp'] < timestamp]
        if filtered_df.empty:  
            raise ValueError(f"No available data before {timestamp}")

        return filtered_df[['pcr_zscore', 'volume_zscore', 'volume_ma_short', 'volume_ma_long']].values[-1].reshape(1, -1)

    
    def get_next_row(self):
        """
        Get the next row of data to simulate real-time data retrieval.
        """
        if self.filtered_df is None or self.filtered_df.empty:
            raise ValueError("No data loaded or data is empty.")

        if not hasattr(self, 'current_index'):
            self.current_index = 0 

        if self.current_index >= len(self.filtered_df):
            raise ValueError("No more data available for the given date.")

        # Get the next row
        next_row = self.filtered_df.iloc[self.current_index]
        self.current_index += 1
        return next_row