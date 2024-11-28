import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import joblib 
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input# type: ignore
from tensorflow.keras.optimizers import Adam# type: ignore
from sklearn.preprocessing import StandardScaler
from pipeline import DataPipeline
import numpy as np
import asyncio
import csv

class Trainer:
    def __init__(self, date, hyperparameters, trend_target_time):
        self.date = date
        self.hyperparameters = hyperparameters
        self.trend_target_time = trend_target_time
        self.data_pipeline = None
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.X_val = None
        self.y_val = None

    def process_data(self):
        """
        Load and preprocess data using the DataPipeline class.
        """
        self.data_pipeline = DataPipeline(
            date=self.date,
            rolling_window=self.hyperparameters['rolling_window'],
            trend_target_time=self.trend_target_time
        )
        try:
            self.data_pipeline.load_data().process_features().validate()
            return self.data_pipeline.get_features_targets()
        except ValueError as e:
            print(e)
            return None, None

    def build_model(self, input_shape):
        """
        Build the model architecture using hyperparameters.
        """
        print("Building model...")
        model = Sequential()
        
        model.add(Input(shape=(input_shape,)))

        for i in range(self.hyperparameters['num_layers']):
            neurons = self.hyperparameters['neurons_per_layer'][i]
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(self.hyperparameters['dropout_rate']))

        model.add(Dense(1, activation='linear'))

        model.compile(optimizer=Adam(learning_rate=self.hyperparameters['learning_rate']),
                    loss='mean_squared_error', 
                    metrics=['mae'])
        
        return model

    def train(self, model_path='model.keras', scaler_path='scalers'):
        """
        Train the model on historical data.
        """
        X, y = self.process_data()
        if X is None or y is None:
            print(f"Skipping {self.date} due to insufficient data.")
            return None
        morning_data_points = min(60, len(X))
        indices = np.arange(morning_data_points)
        np.random.shuffle(indices)
        train_indices = indices[:int(0.8 * len(indices))]
        val_indices = indices[int(0.8 * len(indices)):]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        if len(X_train) < 10 or len(X_val) < 10:
            print(f"Skipping {self.date} due to insufficient training or validation data.")
            return None

        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = self.build_model(X.shape[1])

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        print("Training model...")
        self.model.fit(
            X_train_scaled, y_train_scaled,
            epochs=self.hyperparameters['epochs'],
            batch_size=self.hyperparameters['batch_size'],
            verbose=1
        )
        self.save_model(model_path)

        os.makedirs(scaler_path, exist_ok=True)
        joblib.dump(self.scaler_X, os.path.join(scaler_path, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(scaler_path, 'scaler_y.pkl'))

        return self.model
        

    def predict_real_time(self, X, scaler_path='scalers'):
        """
        Make a prediction using the trained model in real-time mode.
        """
        if self.scaler_X is None or self.scaler_y is None:
            self.load_scalers(scaler_path)

        X_scaled = self.scaler_X.transform(X)
        prediction_scaled = self.model.predict(X_scaled)
        prediction_original_scale = self.scaler_y.inverse_transform(prediction_scaled)
        return prediction_original_scale.flatten()

    
    def save_model(self, file_path='model.keras'):
        self.model.save(file_path)
    
    def load_model(self, file_path='model.keras'):
        if os.path.exists(file_path):
            self.model = tf.keras.models.load_model(file_path)
            
    def load_scalers(self, scaler_path='scalers'):
        """
        Load the scalers used during training.
        """
        self.scaler_X = joblib.load(os.path.join(scaler_path, 'scaler_X.pkl'))
        self.scaler_y = joblib.load(os.path.join(scaler_path, 'scaler_y.pkl'))


async def real_time_prediction(trainer, channel, trend_target_time, data_pipeline, output_dir="predictions"):
    """
    Predict in real-time, every 60 seconds, with Discord notifications, and save predictions to a CSV at the end of the day.
    """
    print("Starting real-time prediction...")
    predictions = []

    while True:
        try:
            new_row = data_pipeline.get_next_row()
            timestamp = new_row['timestamp']
            features = data_pipeline.get_features_for_minute(timestamp)

            prediction = trainer.predict_real_time(features)

            prediction_data = {
                "timestamp": timestamp,
                "prediction_for_20:00_UTC": float(prediction[0])
            }
            predictions.append(prediction_data)

            message = f"FiNNi's prediction at {timestamp} for SPY at {trend_target_time} (UTC): {prediction[0]:.2f}"
            await channel.send(message)

            await asyncio.sleep(60)

        except ValueError as e:
            print(f"No more data available or error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error during real-time prediction: {e}")
            await asyncio.sleep(60)

    output_path = os.path.join(output_dir, trend_target_time, str(data_pipeline.date))
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "predictions.csv")

    try:
        with open(output_file, mode='w', newline='') as file:
            fieldnames = ["timestamp", "prediction_for_20:00_UTC"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(predictions)

        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")