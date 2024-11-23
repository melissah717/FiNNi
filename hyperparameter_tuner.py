import json
from trainer import Trainer

def tune_hyperparameters(date, file="hyperparameters.json"):
    """
    Tune hyperparameters by iterating through combinations from a JSON file.
    """
    print(f"Loading hyperparameters from {hyperparameter_file}")
    with open(file, "r") as f:
        hyperparameter_combinations = json.load(f)

    best_params = None
    best_performance = float("inf")

    print("Starting hyperparameter tuning...")
    for idx, params in enumerate(hyperparameter_combinations):
        print(f"Running combination {idx + 1}/{len(hyperparameter_combinations)}: {params}")

        try:
            trainer = Trainer(date=date, hyperparameters=params)
            val_df = trainer.run(train=True, evaluate=True)

            if val_df is not None:
                loss = trainer.model.evaluate(trainer.X_val, trainer.y_val, verbose=0)[0]
                print(f"Validation Loss for combination {idx + 1}: {loss:.4f}")

                if loss < best_performance:
                    best_performance = loss
                    best_params = params

        except Exception as e:
            print(f"Error with parameters {params} (Combination {idx + 1}): {e}")
            continue  
    print("\nHyperparameter Tuning Completed.")
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation Loss: {best_performance:.4f}")
    return best_params, best_performance
