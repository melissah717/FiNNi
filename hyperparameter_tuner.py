import json
from trainer import Trainer

def tune_hyperparameters(date, file="hyperparameters.json"):
    """
    Tune hyperparameters by iterating through combinations from a JSON file.
    """
    print(f"Loading hyperparameters from {file}")
    with open(file, "r") as f:
        hyperparameter_combinations = json.load(f)

    best_params = None
    best_performance = float("inf")
    results = []
    
    print("Starting hyperparameter tuning...")
    for idx, params in enumerate(hyperparameter_combinations):
        print(f"Running combination {idx + 1}/{len(hyperparameter_combinations)}: {params}")


        try:
            trainer = Trainer(date=date, hyperparameters=params, trend_target_time='14:00:00')
            val_df = trainer.run(train=True, evaluate=True)

            if val_df is not None:
                loss = trainer.model.evaluate(trainer.X_val, trainer.y_val, verbose=0)[0]
                print(f"Validation Loss for combination {idx + 1}: {loss:.4f}")

                results.append((params, loss))

        except Exception as e:
            print(f"Error with parameters {params} (Combination {idx + 1}): {e}")
            continue  
     
    results = sorted(results, key=lambda x: x[1])
    top_3_results = results[:3]       
    print("\nHyperparameter Tuning Completed.")
    for i, (params, loss) in enumerate(top_3_results, start=1):
        print(f"Top {i} Parameters: {params}")
        print(f"Validation Loss: {loss:.4f}")

    return top_3_results

if __name__ == "__main__":
    date = "2024-11-19"
    tune_hyperparameters(date)
