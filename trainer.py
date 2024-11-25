import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import joblib 
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input# type: ignore
from tensorflow.keras.optimizers import Adam# type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pipeline import DataPipeline
import numpy as np
import matplotlib.pyplot as plt


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

        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val)

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
    