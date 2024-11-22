MONGO_URI = "mongodb://localhost:27017" 
DATABASE_NAME = "options_data"
COLLECTION_NAME = "snapshots"

HYPERPARAMETERS = {
    "skew_threshold": 1.5,
    "rolling_window": 60,
    "trend_window": 20,
    "neurons_per_layer": [128, 64, 32],
    "dropout_rate": 0.2,
    "num_layers": 3,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.01,
}