from trainer import Trainer
from pipeline import DataPipeline
import logging
import discord
import asyncio
from dotenv import load_dotenv
import os
import pandas as pd
from utils import display_welcome
from datetime import timedelta

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))

# Define best hyperparameters
BEST_PARAMS = {
    "skew_threshold": 1.25,
    "rolling_window": 30,
    "neurons_per_layer": [64, 32],
    "dropout_rate": 0.1,
    "num_layers": 2,
    "batch_size": 16,
    "epochs": 200,
    "learning_rate": 0.00001
}

logging.basicConfig(level=logging.INFO)

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    await main()

async def main(batch_mode=False):
    logging.info("Starting main workflow...")

    # These are the dates for training
    training_dates = ["2024-11-14", "2024-11-13", "2024-11-15", "2024-11-19", "2024-11-20", "2024-11-21", "2024-11-22", "2024-11-25", "2024-11-26"]
    test_date = "2024-11-27"
    trend_target_time = "20:00:00"
    model_path = "model.keras"

    # Training phase
    for date in training_dates:
        trainer = Trainer(date, BEST_PARAMS, trend_target_time)
        trainer.train(model_path)

    # Load the trained model for testing
    trainer = Trainer(test_date, BEST_PARAMS, trend_target_time)
    trainer.load_model(model_path)

    # Load the test data for batch or real-time prediction
    prediction_mode = not batch_mode  # If batch mode is False, use prediction mode
    data_pipeline = DataPipeline(test_date, rolling_window=BEST_PARAMS['rolling_window'], trend_target_time=trend_target_time, prediction_mode=prediction_mode)
    data_pipeline.load_data().process_features()

    # Get Discord channel
    channel = client.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found.")
        return

    if batch_mode:
        print("Running batch predictions for the entire dataset of the test date...")
        predictions = []
        for _, row in data_pipeline.filtered_df.iterrows():
            features = row[['pcr_zscore', 'volume_zscore', 'volume_ma_short', 'volume_ma_long', 'underlying_price']].values.reshape(1, -1)
            prediction = trainer.predict_real_time(features)
            predictions.append({
                'timestamp': row['timestamp'],
                'predicted_at_target_time': prediction[0]
            })

        predictions_df = pd.DataFrame(predictions)
        output_file = f'predictions/{test_date}.csv'
        predictions_df.to_csv(output_file, index=False)
        print(f"Batch predictions saved to {output_file}")

    else:
        print("Running in real-time prediction mode...")
        custom_emoji_name = os.getenv("CUSTOM_EMOJI_NAME")
        custom_emoji_id = os.getenv("CUSTOM_EMOJI_ID")
        is_first_valid_row = False

        while True:
            try:
                # Continuously get the next available row that is the closest to the current adjusted time
                next_row = data_pipeline.get_next_row(rolling_window=BEST_PARAMS['rolling_window'])
                if next_row is None:
                    await asyncio.sleep(60)  # Wait for a minute before retrying
                    continue

                timestamp = next_row['timestamp']
                try:
                    features = data_pipeline.get_features_for_minute(timestamp)
                except ValueError as e:
                    print(f"Skipping timestamp {timestamp} due to insufficient data.")
                    await asyncio.sleep(60)  # Wait for 60 seconds before retrying
                    continue

                # Make prediction
                prediction = trainer.predict_real_time(features)
                message_text = (
                    f"Using data from {timestamp}, FiNNi predicts SPY will be at {trend_target_time} (UTC): {prediction[0]:.2f}"
                )

                custom_emoji = f"<:{custom_emoji_name}:{custom_emoji_id}>"
                message_text += f" {custom_emoji}"
                await channel.send(message_text)

                # Ensure proper sleep in between rows for real-time emulation
                if is_first_valid_row:
                    await asyncio.sleep(60)
                else:
                    is_first_valid_row = True

            except ValueError as e:
                print(f"No more data available or error: {e}")
                break

if __name__ == "__main__":
    display_welcome()
    asyncio.run(client.start(TOKEN))
