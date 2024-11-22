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
