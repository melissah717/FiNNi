import json
from trainer import Trainer
from utils import plot_hyperparameter_performance
import numpy as np
import os

def tune_hyperparameters(dates, file="hyperparameters.json"):
    """
    Tune hyperparameters by iterating through combinations from a JSON file.
    """
    print(f"Loading hyperparameters from {file}")
    with open(file, "r") as f:
        hyperparameter_combinations = json.load(f)

    results = []

    print("Starting hyperparameter tuning...")
    for idx, params in enumerate(hyperparameter_combinations):
        print(f"Running combination {idx + 1}/{len(hyperparameter_combinations)}: {params}")
        mae_list = []
        mse_list = []

        # Iterate over the list of dates for each hyperparameter combination
        for date in dates:
            try:
                # Create a new Trainer instance for every date and combination to ensure no reuse of the model
                trainer = Trainer(date=date, hyperparameters=params, trend_target_time='20:00:00')

                # Assign a unique temporary model path for each hyperparameter combination and date
                model_path = f"temp_models/model_combination_{idx + 1}_date_{date}.keras"
                scaler_path = f"temp_scalers/scaler_combination_{idx + 1}_date_{date}"

                # Train a fresh model on the specified date
                trainer.train(model_path=model_path, scaler_path=scaler_path)

                # Check if the model file was saved correctly
                if os.path.exists(model_path):
                    # Evaluate the model on the validation set
                    if trainer.X_val is not None and trainer.y_val is not None:
                        loss, mae = trainer.model.evaluate(trainer.X_val, trainer.y_val, verbose=0)
                        mse = loss  # Since MSE is the loss function used
                        mae_list.append(mae)
                        mse_list.append(mse)
                        print(f"Date: {date} - MAE: {mae:.4f}, MSE: {mse:.4f}")

                    # Remove the temporary model to save space
                    os.remove(model_path)

                # Remove scaler files if they exist
                if os.path.exists(f"{scaler_path}/scaler_X.pkl"):
                    os.remove(f"{scaler_path}/scaler_X.pkl")
                if os.path.exists(f"{scaler_path}/scaler_y.pkl"):
                    os.remove(f"{scaler_path}/scaler_y.pkl")

            except Exception as e:
                print(f"Error with date {date} and parameters {params} (Combination {idx + 1}): {e}")
                continue

        # Calculate average MAE and MSE for the hyperparameter combination across all training dates
        avg_mae = np.mean(mae_list) if mae_list else float("inf")
        avg_mse = np.mean(mse_list) if mse_list else float("inf")
        results.append((params, avg_mae, avg_mse))

    # Sort results by average MAE to determine the top performers
    results = sorted(results, key=lambda x: x[1])
    top_3_results = results[:3]

    # Display the top 3 results
    print("\nHyperparameter Tuning Completed.")
    for i, (params, avg_mae, avg_mse) in enumerate(top_3_results, start=1):
        print(f"Top {i} Parameters: {params}")
        print(f"Average MAE: {avg_mae:.4f}, Average MSE: {avg_mse:.4f}")

    # Plot the performance of each hyperparameter combination
    plot_hyperparameter_performance(results)

    return top_3_results


if __name__ == "__main__":
    dates = ["2024-11-13", "2024-11-14", "2024-11-15", "2024-11-19", "2024-11-20", "2024-11-21", "2024-11-22"]
    tune_hyperparameters(dates)
