from utils import load

class DataPipeline:
    def __init__(self, date, rolling_window, trend_window):
        self.date = date
        self.rolling_window = rolling_window
        self.trend_window = trend_window
        self.filtered_df = None
        
    def load_data(self):
        print(f"Loading {self.date}")
        self.filtered_df = load(self.date)
        if self.filtered_df.empty or len(self.filtered_df) < 30:
            raise ValueError("No data available")
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
        df = df.dropna()
        self.filtered_df = df
        return self
    
    def validate(self):
        print("Validating")
        if len(self.filtered_df) < self.trend_window:
            raise ValueError(f"Not enough data for trend window")
        return self
    
    def get_features_targets(self):
        X = self.filtered_df[['pcr_zscore', 'volume_zscore']].iloc[:-self.trend_window].values
        y = self.filtered_df['underlying_price'].iloc[self.trend_window:].values.reshape(-1, 1)
        return X, y