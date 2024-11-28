from modes.batch import run_batch
from modes.live import run_live
import asyncio
import logging
from utils import display_welcome

if __name__ == "__main__":
    display_welcome()
    MODE = (
        input("Enter mode (1 for batch training / 2 for real-time predicting): ")
        .strip()
        .lower()
    )
    model_path = "model.keras"
    if MODE == "1":
        train_dates = [
            "2024-11-13",
            "2024-11-14",
            "2024-11-15",
            "2024-11-18",
            "2024-11-19",
            "2024-11-20",
            "2024-11-21",
            "2024-11-22",
            "2024-11-25",
            "2024-11-26"
        ]
        test_date = "2024-11-27"
        target_time = "20:30"
        hyperparameters = {
            "rolling_window": 60,
            "num_layers": 2,
            "neurons_per_layer": [64, 32],
            "dropout_rate": 0.1,
            "learning_rate": 0.00001,
            "epochs": 100,
            "batch_size": 8,
        }
        run_batch(train_dates, test_date, hyperparameters, target_time, model_path)

    elif MODE == "2":
        logging.info("Live prediction not implemented yet!")
        # target_time = "20:00"
        # date = "2024-11-28"
        # hyperparameters = {"rolling_window": 20}
        # trainer = Trainer(hyperparameters)
        # data_pipeline = DataPipeline(
        #     date, hyperparameters["rolling_window"], target_time
        # )
        # asyncio.run(run_live(trainer, data_pipeline, target_time))
    else:
        print("Invalid mode. Please choose 1 or 2.")
