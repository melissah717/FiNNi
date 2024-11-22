from process import process_and_save_data
from model import evolve
from graph import plot_vs, plot_training_validation_loss
from config import HYPERPARAMETERS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    date = "2024-11-19"

    print(f"Processing data for {date}...")
    X, y, filtered_df = process_and_save_data(
        date,
        HYPERPARAMETERS['skew_threshold'],
        HYPERPARAMETERS['rolling_window'],
        HYPERPARAMETERS['trend_window']
    )

    if X is None or y is None:
        print(f"Skipping {date} due to insufficient data.")
        return

    print("Training model...")
    model, history, scaler_y, X_val, y_val = evolve(X, y, HYPERPARAMETERS)

    predictions = model.predict(X_val)
    predictions_original_scale = scaler_y.inverse_transform(predictions)
    y_val_original_scale = scaler_y.inverse_transform(y_val)

    val_df = filtered_df.iloc[-len(y_val):].copy()
    val_df['Actual'] = y_val_original_scale.flatten()
    val_df['Predicted'] = predictions_original_scale.flatten()
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Training MAE: {history.history['mae'][-1]:.4f}")
    print(f"Final Validation MAE: {history.history['val_mae'][-1]:.4f}")

    print("Plotting results...")
    # plot_vs(val_df)
    # plot_training_validation_loss(history)
    print(val_df.head(20))

if __name__ == "__main__":
    main()