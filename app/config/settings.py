import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ehr_readmission_db")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "[]")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()