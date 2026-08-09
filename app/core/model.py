# app/core/model.py
import joblib
import numpy as np
import os
from app.config.settings import settings
from app.core.features import get_feature_names
import logging

logger = logging.getLogger(__name__)

class ReadmissionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = get_feature_names()
        self.metadata = {}
        self.version = "1.0.0"
        self.load()

    def load(self):
        model_path = settings.MODEL_PATH
        
        # Try to load version 2 first
        model_v2 = os.path.join(model_path, "xgboost_model_v2.pkl")
        scaler_v2 = os.path.join(model_path, "scaler_v2.pkl")
        
        if os.path.exists(model_v2) and os.path.exists(scaler_v2):
            self.model = joblib.load(model_v2)
            self.scaler = joblib.load(scaler_v2)
            self.version = "2.0.0"
            logger.info("Loaded model version 2.0.0")
        else:
            # Fall back to version 1
            self.model = joblib.load(os.path.join(model_path, "xgboost_model.pkl"))
            self.scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
            self.version = "1.0.0"
            logger.info("Loaded model version 1.0.0")
        
        try:
            with open(os.path.join(model_path, "model_metadata.json"), "r") as f:
                import json
                self.metadata = json.load(f)
        except:
            self.metadata = {"version": self.version}

    def predict(self, features):
        """Make prediction for a single patient"""
        scaled = self.scaler.transform(np.array(features).reshape(1, -1))
        prob = self.model.predict_proba(scaled)[0, 1]
        return float(prob)

    def predict_batch(self, features_list):
        """Make predictions for multiple patients"""
        scaled = self.scaler.transform(np.array(features_list))
        probs = self.model.predict_proba(scaled)[:, 1]
        return probs.tolist()
    
    def get_version(self):
        return self.version