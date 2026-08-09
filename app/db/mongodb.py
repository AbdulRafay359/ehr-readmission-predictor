from pymongo import MongoClient
from app.config.settings import settings

client = None
db = None

def connect_to_mongo():
    global client, db
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    return db

def get_db():
    if db is None:
        connect_to_mongo()
    return db