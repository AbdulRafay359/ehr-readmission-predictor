# calibrate_thresholds.py
from app.db.mongodb import connect_to_mongo
from app.core.model import ReadmissionModel
from app.core.features import extract_features
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score

def calibrate_thresholds():
    db = connect_to_mongo()
    
    # Load the new model
    import joblib
    import os
    from app.config.settings import settings
    
    model_path = os.path.join(settings.MODEL_PATH, "xgboost_model_v2.pkl")
    scaler_path = os.path.join(settings.MODEL_PATH, "scaler_v2.pkl")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Using model version 2.0.0")
    else:
        # Fall back to original model
        model = joblib.load(os.path.join(settings.MODEL_PATH, "xgboost_model.pkl"))
        scaler = joblib.load(os.path.join(settings.MODEL_PATH, "scaler.pkl"))
        print("Using model version 1.0.0")
    
    # Get balanced sample
    positive = list(db.diabetes_patients_combined.find({"readmitted": "<30"}).limit(500))
    negative = list(db.diabetes_patients_combined.find({"readmitted": "NO"}).limit(500))
    
    all_patients = positive + negative
    y_true = [1 if p.get('readmitted') == '<30' else 0 for p in all_patients]
    
    # Get predictions
    y_pred_proba = []
    for patient in all_patients:
        features = extract_features(patient)
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        risk = model.predict_proba(scaled_features)[0, 1]
        y_pred_proba.append(risk)
    
    y_pred_proba = np.array(y_pred_proba)
    
    # Find optimal thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    print("\nThreshold Calibration")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Predictions >':<15}")
    print("-" * 70)
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        pred_count = sum(y_pred)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"{thresh:.2f}        {precision:.4f}     {recall:.4f}     {f1:.4f}     {pred_count:>4}/{len(y_pred)}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
    
    print("-" * 70)
    print(f"\nBest threshold for F1: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Define risk levels
    risk_levels = {
        "very_low": {"threshold": 0.0, "label": "Very Low Risk"},
        "low": {"threshold": 0.15, "label": "Low Risk"},
        "medium": {"threshold": 0.3, "label": "Medium Risk"},
        "high": {"threshold": 0.55, "label": "High Risk"},
        "very_high": {"threshold": 0.75, "label": "Very High Risk"}
    }
    
    print("\n" + "=" * 70)
    print("Recommended Risk Level Thresholds")
    print("=" * 70)
    for level, info in risk_levels.items():
        y_pred_level = (y_pred_proba >= info["threshold"]).astype(int)
        count = sum(y_pred_level)
        pct = (count / len(y_true)) * 100
        print(f"{info['label']:>18}: >= {info['threshold']:.2f}  ({count:>4} patients, {pct:.1f}%)")
    
    # Save thresholds
    thresholds_config = {
        "risk_levels": risk_levels,
        "optimal_threshold": best_threshold,
        "optimal_metrics": best_metrics,
        "model_version": "2.0.0",
        "calibration_date": pd.Timestamp.now().isoformat()
    }
    
    with open("models/risk_thresholds.json", "w") as f:
        json.dump(thresholds_config, f, indent=2)
    
    print(f"\nThresholds saved to: models/risk_thresholds.json")
    
    return thresholds_config

if __name__ == "__main__":
    import pandas as pd
    calibrate_thresholds()