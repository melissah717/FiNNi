from trainer import Trainer
from hyperparameter_tuner import tune_hyperparameters
    
def main():
    date = "2024-11-20"
    
    # uncomment this block to run hyperparameter tuning
    best_params, best_performance = tune_hyperparameters(date)
    print("\nFinal Results:")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Loss: {best_performance:.4f}")
    
    ## OR 
    
    # Run a single date with default hyperparameters
    trainer = Trainer(date, best_params)
    val_df = trainer.run(train=True, evaluate=True)
    if val_df is not None:
        print(val_df.head())

if __name__ == "__main__":
    main()