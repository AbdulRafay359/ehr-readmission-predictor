import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
import joblib
import json
import os
from app.db.mongodb import connect_to_mongo
from app.core.features import extract_features, get_feature_names

def load_data_from_mongo():
    db = connect_to_mongo()
    cursor = db.diabetes_patients_combined.find({})
    records = list(cursor)
    X = []
    y = []
    for rec in records:
        feat = extract_features(rec)
        target = rec.get("readmitted_30d", 0)
        X.append(feat)
        y.append(target)
    return np.array(X), np.array(y)

def train():
    X, y = load_data_from_mongo()
    print(f"Loaded {len(X)} samples, {len(X[0])} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="binary:logistic",
        random_state=42,
        scale_pos_weight=0.3,
        early_stopping_rounds=20,
        eval_metric="auc"
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "f1_score": f1_score(y_test, y_pred)
    }
    print("Validation metrics:", metrics)

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/feature_columns.pkl", "wb") as f:
        import pickle
        pickle.dump(get_feature_names(), f)

    metadata = {
        "version": "1.0.0",
        "training_date": pd.Timestamp.now().isoformat(),
        "metrics": metrics,
        "n_features": X.shape[1],
        "n_samples": len(X)
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete. Artifacts saved to ./models/")

if __name__ == "__main__":
    train()