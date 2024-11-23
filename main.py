from trainer import Trainer
from hyperparameter_tuner import tune_hyperparameters
import json

def main():
    date = "2024-11-21"
    
    # uncomment this block to run hyperparameter tuning
    # best_params, best_performance = tune_hyperparameters(date)
    # print("\nFinal Results:")
    # print(f"Best Hyperparameters: {best_params}")
    # print(f"Best Validation Loss: {best_performance:.4f}")
    
    ## OR 
    
    # Run a single date with default hyperparameters
    with open("hyperparameters.json", "r") as f:
        hyperparameters_list = json.load(f) 
        single_params = hyperparameters_list[0] 
    trainer = Trainer(date, single_params)
    val_df = trainer.run(train=True, evaluate=True)
    if val_df is not None:
        # if "options_data" in val_df.columns:
        #     val_df = val_df.drop(columns=["options_data"])
        
        # # Save the resulting DataFrame to a CSV file
        # val_df.to_csv(f"results-{date}.csv", index=False)
        print(val_df.tail())

if __name__ == "__main__":
    main()