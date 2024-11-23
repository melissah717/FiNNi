"""
utils.py

Modules:
- load: Fetch data from MongoDB for a specific date.
- generate_hyperparameters: Generate combinations of hyperparameters.
- load_hyperparameters: Load hyperparameters from a JSON file.
- plot_vs: Plot actual vs predicted values.
- plot_training_validation_loss: Visualize training and validation loss.

Author: Melissa Ho
Date: 2024-11-23
"""

import matplotlib.pyplot as plt
import itertools
import json
import os
from dotenv import load_dotenv
import pandas as pd
from pymongo import MongoClient
from typing import List, Dict, Optional

load_dotenv()

def load(date: str) -> pd.DataFrame:
    """
    Connect to DB and fetch snapshots for a specific date
    
    Args:
        date(str): "YYYY-MM-DD" format
        
    Returns:
        pd.DataFrame: df containing snapshots for a given date
    """
    
    try:
        parsed_date = pd.to_datetime(date).date()
    except ValueError:
        raise ValueError(f"Invalid date formate: {date}, please use YYYY-MM-DD")
    client = MongoClient(os.getenv("MONGO_URI"))
    
    try:
        db = client[os.getenv("DATABASE_NAME")]
        collection = db[os.getenv("COLLECTION_NAME")]
        snapshots = list(collection.find())
        df = pd.DataFrame(snapshots)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df[df['timestamp'].dt.date == parsed_date]
    finally:
        client.close()
    
    

def generate_hyperparameters(grid):
    """
    Generates all different combinations of hyperparameters
    
    sample usage
    param_grid = {
        "skew_threshold": [1, 1.5],
        "rolling_window": [60, 120],
        "trend_window": [30, 60],
        "neurons_per_layer": [[128, 64, 32]],
        "dropout_rate": [0.2],
        "num_layers": [3],
        "batch_size": [16],
        "epochs": [50, 100],
        "learning_rate": [0.001]
    }     
    generate_hyperparameters(param_grid)
    
    Output:
        'hyperparameters.json' in root directory
    
    """

    keys, values = zip(*grid.items())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    output_file = "hyperparameters.json"
    with open(output_file, 'w') as f:
        json.dump(param_combos, f, indent=4)   
        

def load_hyperparameters(file_path: str="hyperparameters.json") -> Optional[List[Dict]]:
    """
    Loads hyperparameters from the JSON File generated
    
    Args:
        file_path (str): path to json file
    Returns:
        Optional[List[Dict]]: a list of hyperparameter combos
    Raises:
        FileNotFoundError: if file doesn't exist
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"file not found at: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid json format in file: {file_path}") from e
    
    
def plot_vs(val_df):
    """
    Plot scatterplot of actual vs predicted values, without time axis
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(val_df['Actual'], val_df['Predicted'], alpha=0.7)
    plt.plot([val_df['Actual'].min(), val_df['Actual'].max()], 
             [val_df['Actual'].min(), val_df['Actual'].max()], 
             color='green', linestyle='--', label="Perfect Prediction")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted underlying prices')
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()
    
def plot_training_validation_loss(history):
    """
    Plot training and validation loss over epochs
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