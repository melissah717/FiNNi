import asyncio
from trainer import Trainer
from pipeline import DataPipeline
import logging
import joblib
import os


async def run_live(trainer, data_pipeline, target_time):
    while True:
        try:
            next_row = data_pipeline.get_next_row(data_pipeline.rolling_window)
            if next_row is None:
                logging.info()
                await asyncio.sleep(60)
                continue

            timestamp = next_row['timestamp']
            features = data_pipeline.get_features_for_minute(timestamp)
            if features is None:
                await asyncio.sleep(60)
                continue

            prediction = trainer.predict(features)
            print(f"Prediction for {target_time} at {timestamp}: {prediction}")
            await asyncio.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            break

def load_scalers(self, scaler_path='scalers'):
    """
    Load the scalers used during training.
    """
    self.scaler_X = joblib.load(os.path.join(scaler_path, 'scaler_X.pkl'))
    self.scaler_y = joblib.load(os.path.join(scaler_path, 'scaler_y.pkl'))
    print("Scalers loaded successfully.")