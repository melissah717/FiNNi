from trainer import Trainer
# from hyperparameter_tuner import tune_hyperparameters
from pipeline import DataPipeline
import logging
import discord
from discord.ext import tasks
import asyncio
from dotenv import load_dotenv
import os
from utils import display_welcome

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))
BEST_PARAMS = {
    "skew_threshold": 1.5,
    "rolling_window": 60,
    "neurons_per_layer": [64, 32],
    "dropout_rate": 0.05,
    "num_layers": 2,
    "batch_size": 16,
    "epochs": 300,
    "learning_rate": 0.000001
}

logging.basicConfig(level=logging.INFO)

# Set up Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"logged in as {client.user}")
    await main()
    
async def main():
    logging.basicConfig(level=logging.INFO)
    
    #these are the dates for training!
    dates = ["2024-11-13", "2024-11-14", "2024-11-15", "2024-11-19", "2024-11-20", "2024-11-21"]
    test_date = "2024-11-22"
    
    # uncomment this block to run hyperparameter tuning
    # best_params, best_performance = tune_hyperparameters(date)
    # print("\nFinal Results:")
    # print(f"Best Hyperparameters: {best_params}")
    # print(f"Best Validation Loss: {best_performance:.4f}")

    trend_target_time = "20:00:00"
    model_path = "model.keras"

    for date in dates:
        trainer = Trainer(date, BEST_PARAMS, trend_target_time)
        trainer.train(model_path)

    trainer = Trainer(test_date, BEST_PARAMS, trend_target_time)
    trainer.load_model(model_path)

    data_pipeline = DataPipeline(test_date, rolling_window=BEST_PARAMS['rolling_window'], trend_target_time=trend_target_time)
    data_pipeline.load_data().process_features()


    channel = client.get_channel(int(CHANNEL_ID))
    if not channel:
        print("Channel not found")
        return

    print("Starting real-time prediction simulation...")
    is_first_valid_row = False
    custom_emoji_name = os.getenv("CUSTOM_EMOJI_NAME")
    custom_emoji_id = os.getenv("CUSTOM_EMOJI_ID")
    
    
    while True:
        try:
            next_row = data_pipeline.get_next_row()
            timestamp = next_row['timestamp']

            try:
                features = data_pipeline.get_features_for_minute(timestamp)
            except ValueError as e:
                print(f"Skipping timestamp {timestamp} due to insufficient data.")
                continue
            prediction = trainer.predict_real_time(features)
            message_text = f"FiNNi's guess at {timestamp} for SPY at {trend_target_time} (UTC): {prediction[0]:.2f}"

            custom_emoji = f"<:{custom_emoji_name}:{custom_emoji_id}>"
            message_text += f" {custom_emoji}"
            await channel.send(message_text)

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