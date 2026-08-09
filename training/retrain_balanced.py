# training/retrain_balanced.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
import joblib
import json
from app.db.mongodb import connect_to_mongo
from app.core.features import extract_features, get_feature_names

def load_data_from_mongo():
    db = connect_to_mongo()
    cursor = db.diabetes_patients_combined.find({})
    records = list(cursor)
    X = []
    y = []
    print(f"Loading data from MongoDB...")
    for rec in records:
        feat = extract_features(rec)
        target = rec.get("readmitted_30d", 0)
        X.append(feat)
        y.append(target)
    return np.array(X), np.array(y)

def retrain_balanced():
    X, y = load_data_from_mongo()
    print(f"Loaded {len(X)} samples, {len(X[0])} features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Calculate scale_pos_weight
    scale_pos_weight = counts[0] / counts[1] if counts[1] > 0 else 1
    print(f"Scale_pos_weight: {scale_pos_weight:.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with multiple techniques to handle imbalance
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        objective="binary:logistic",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric="auc"
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "f1_score": float(f1_score(y_test, y_pred))
    }
    
    print("\nValidation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Show prediction distribution on test set
    print(f"\nPrediction distribution on test set:")
    print(f"  Predicted 0: {len(y_pred) - sum(y_pred)}")
    print(f"  Predicted 1: {sum(y_pred)}")
    print(f"  Actual 0: {len(y_test) - sum(y_test)}")
    print(f"  Actual 1: {sum(y_test)}")
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model_v2.pkl")
    joblib.dump(scaler, "models/scaler_v2.pkl")
    with open("models/feature_columns_v2.pkl", "wb") as f:
        import pickle
        pickle.dump(get_feature_names(), f)
    
    # Save model info - CONVERT ALL NUMPY TYPES TO PYTHON TYPES
    metadata = {
        "version": "2.0.0",
        "training_date": pd.Timestamp.now().isoformat(),
        "metrics": metrics,
        "n_features": int(X.shape[1]),
        "n_samples": int(len(X)),
        "scale_pos_weight": float(scale_pos_weight),
        "class_distribution": {int(k): int(v) for k, v in zip(unique, counts)}
    }
    
    with open("models/model_metadata_v2.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining complete. Artifacts saved to ./models/")
    print(f"New model saved as: xgboost_model_v2.pkl")

if __name__ == "__main__":
    retrain_balanced()