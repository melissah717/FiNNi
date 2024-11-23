import matplotlib.pyplot as plt
import itertools
import json
import pandas as pd
from pymongo import MongoClient
from config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME

def load(date):
    """
    Connect to DB and fetch snapshots
    """
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    snapshots = list(collection.find())
    df = pd.DataFrame(snapshots)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    client.close()
    
    
    return df[df['timestamp'].dt.date == pd.to_datetime(date).date()]


#TODO make this dynamic
def generate_hyperparameters(grid):
    """
    Generates a json file with all different combinations of parameters.
    FINNI loops through these parameters and displays the ones with the best results
    """

    keys, values = zip(*param_grid.items())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_file = "hyperparameters.json"
    with open(output_file, 'w') as f:
        json.dump(param_combos, f, indent=4)   
        
        
param_grid = {
    "skew_threshold": [1, 1.5],
    "rolling_window": [ 60, 120],
    "trend_window": [ 30, 60],
    "neurons_per_layer": [[128, 64, 32]],
    "dropout_rate": [0.2],
    "num_layers": [3],
    "batch_size": [16],
    "epochs": [50, 100],
    "learning_rate": [0.001]
}     
generate_hyperparameters(param_grid)

def load_hyperparameters(file_path="hyperparameters.json"):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Hyperparameter file not found: {file_path}")
        return None
    
    
def plot_vs(val_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(val_df['Actual'], val_df['Predicted'], alpha=0.7)
    plt.plot([val_df['Actual'].min(), val_df['Actual'].max()], 
             [val_df['Actual'].min(), val_df['Actual'].max()], 
             color='red', linestyle='--', label="Perfect Prediction")
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()
    
def plot_training_validation_loss(history):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()