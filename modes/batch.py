from trainer import Trainer
from pipeline import DataPipeline
import pandas as pd

def run_batch(dates, test, hyperparameters, target_time, model_path='model.keras', scaler_path='scalers'):
    trainer = Trainer(hyperparameters)

    for date in dates:
        pipeline = DataPipeline(date, hyperparameters['rolling_window'], target_time)
        pipeline.load_data().process_features()
        X, y = pipeline.get_features_targets()
        trainer.train(X,y)

    test_pipeline = DataPipeline(test, hyperparameters['rolling_window'], target_time)
    test_pipeline.load_data().process_features()
    X_test, _= test_pipeline.get_features_targets()

    trainer.load_model(model_path)
    trainer.load_scalers(scaler_path)
    predictions = trainer.predict(X_test)

    timestamps = test_pipeline.filtered_df['timestamp']
    prediction_results = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': predictions
    })

    print(prediction_results)
    return prediction_results