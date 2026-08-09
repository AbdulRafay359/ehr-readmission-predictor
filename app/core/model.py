import joblib
import numpy as np
import os
from app.config.settings import settings
from app.core.features import get_feature_names

class ReadmissionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = get_feature_names()
        self.metadata = {}
        self.load()

    def load(self):
        model_path = settings.MODEL_PATH
        self.model = joblib.load(os.path.join(model_path, "xgboost_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
        # feature columns already defined
        try:
            with open(os.path.join(model_path, "model_metadata.json"), "r") as f:
                import json
                self.metadata = json.load(f)
        except:
            self.metadata = {"version": "unknown"}

    def predict(self, features):
        scaled = self.scaler.transform(np.array(features).reshape(1, -1))
        prob = self.model.predict_proba(scaled)[0, 1]
        return float(prob)

    def predict_batch(self, features_list):
        scaled = self.scaler.transform(np.array(features_list))
        probs = self.model.predict_proba(scaled)[:, 1]
        return probs.tolist()