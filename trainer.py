import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input# type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pipeline import DataPipeline

class Trainer:
    def __init__(self, date, hyperparameters):
        self.date = date
        self.hyperparameters = hyperparameters
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
            trend_window=self.hyperparameters['trend_window']
        )
        try:
            self.data_pipeline.load_data().process_features().validate()
            return self.data_pipeline.get_features_targets()
        except ValueError as e:
            print(e)
            return None, None

    def build_model(self, input_shape):
        """
        Build the model architecture.
        """
        print("Building model...")
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def run(self, train=True, evaluate=True):
        """
        Main workflow: process data, train model, and evaluate results.
        """
        X, y = self.process_data()
        if X is None or y is None:
            print(f"Skipping {self.date} due to insufficient data.")
            return None


        self.model = self.build_model(X.shape[1])

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X_train, self.X_val, y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if train:
            print("Training model...")
            self.model.fit(
                X_train, y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=self.hyperparameters['epochs'],
                batch_size=self.hyperparameters['batch_size'],
                verbose=1
            )

        if evaluate:
            print("Evaluating model...")
            predictions = self.model.predict(self.X_val)
            predictions_original_scale = self.scaler_y.inverse_transform(predictions)
            y_val_original_scale = self.scaler_y.inverse_transform(self.y_val)

            val_df = self.data_pipeline.filtered_df.iloc[-len(self.y_val):].copy()
            val_df['Actual'] = y_val_original_scale.flatten()
            val_df['Predicted'] = predictions_original_scale.flatten()

            loss = self.model.evaluate(self.X_val, self.y_val, verbose=0)
            print(f"Validation Loss: {loss[0]:.4f}")
            print(f"Validation MAE: {loss[1]:.4f}")

            return val_df
