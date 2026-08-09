# get_risk_patients.py
import sys
import os
import json
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db.mongodb import connect_to_mongo
from app.core.model import ReadmissionModel
from app.core.features import extract_features
import numpy as np
import joblib
from app.config.settings import settings


def get_patients_by_risk_level():
    """
    Get a random sample: 2 Low Risk, 1 Moderate Risk, 2 High Risk patients.
    Selection is randomized within each category, so re-running this
    returns a different set of 5 patients (as long as each category has
    more candidates than requested).
    """
    db = connect_to_mongo()

    # Load the model
    model_path = os.path.join(settings.MODEL_PATH, "xgboost_model_v2.pkl")
    scaler_path = os.path.join(settings.MODEL_PATH, "scaler_v2.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Using model version 2.0.0")
    else:
        model = joblib.load(os.path.join(settings.MODEL_PATH, "xgboost_model.pkl"))
        scaler = joblib.load(os.path.join(settings.MODEL_PATH, "scaler.pkl"))
        print("Using model version 1.0.0")

    # Load thresholds
    try:
        with open("models/risk_thresholds.json", "r") as f:
            thresholds = json.load(f)
            risk_levels = thresholds["risk_levels"]
            print("Using calibrated thresholds")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        # Fallback thresholds
        risk_levels = {
            "very_low": {"threshold": 0.0, "label": "Very Low Risk"},
            "low": {"threshold": 0.15, "label": "Low Risk"},
            "medium": {"threshold": 0.30, "label": "Medium Risk"},
            "high": {"threshold": 0.55, "label": "High Risk"},
            "very_high": {"threshold": 0.75, "label": "Very High Risk"}
        }
        print("Using default thresholds")

    high_cutoff = risk_levels["high"]["threshold"]
    moderate_cutoff = risk_levels["medium"]["threshold"]
    low_cutoff = risk_levels["low"]["threshold"]

    # Get a large sample of patients
    print("\nScanning patients...")
    all_patients = list(db.diabetes_patients_combined.find({}).limit(5000))
    print(f"Scanned {len(all_patients)} patients")

    # Categorize patients
    risk_categories = {
        "low": [],
        "moderate": [],
        "high": []
    }

    skipped = 0
    for patient in all_patients:
        try:
            features = extract_features(patient)
            scaled_features = scaler.transform(np.array(features).reshape(1, -1))
            risk = model.predict_proba(scaled_features)[0, 1]

            patient_id = str(patient.get('encounter_id'))
            actual_readmitted = patient.get('readmitted', 'Unknown')

            entry = {
                "patient_id": patient_id,
                "risk": float(risk),
                "actual_readmitted": actual_readmitted,
                "age": patient.get('age', 'Unknown'),
                "gender": patient.get('gender', 'Unknown')
            }

            # Determine risk level based on thresholds
            if risk >= high_cutoff:
                risk_categories["high"].append(entry)
            elif risk >= moderate_cutoff:
                risk_categories["moderate"].append(entry)
            elif risk < low_cutoff:
                risk_categories["low"].append(entry)
            # patients strictly between low_cutoff and moderate_cutoff are
            # a gray zone and intentionally left out, same as the original.

        except Exception as e:
            skipped += 1
            continue

    if skipped:
        print(f"Skipped {skipped} patients due to feature/prediction errors")

    # Randomly sample the requested count from each category instead of
    # always taking the highest/lowest scores — re-running this script
    # gives a different set of patients each time.
    def sample(category, count):
        pool = risk_categories[category]
        if len(pool) <= count:
            return pool
        return random.sample(pool, count)

    selected = {
        "low": sample("low", 2),        # 2 low risk
        "moderate": sample("moderate", 1),  # 1 moderate risk
        "high": sample("high", 2)        # 2 high risk
    }

    # Print results
    print("\n" + "=" * 70)
    print("RANDOMLY SELECTED PATIENTS BY RISK LEVEL")
    print("=" * 70)

    all_ids = {}

    for level, patients in selected.items():
        print(f"\n{level.upper()} RISK PATIENTS ({len(patients)} selected):")
        print("-" * 50)

        if not patients:
            print(f"  No candidates available in the '{level}' category.")

        ids = []
        for idx, p in enumerate(patients, 1):
            print(f"  {idx}. Patient ID: {p['patient_id']}")
            print(f"     Risk Score: {p['risk']:.4f}")
            print(f"     Actual Readmitted: {p['actual_readmitted']}")
            print(f"     Age: {p['age']}, Gender: {p['gender']}")
            print()
            ids.append(p['patient_id'])

        all_ids[level] = ids

    # Print the exact IDs for API testing
    print("\n" + "=" * 70)
    print("PATIENT IDs FOR API TESTING")
    print("=" * 70)
    print("\nCopy these IDs for your API request:")
    print()

    # Format as JSON for easy copying
    result = {
        "patient_ids": (
            all_ids.get("low", []) +
            all_ids.get("moderate", []) +
            all_ids.get("high", [])
        )
    }

    print(json.dumps(result, indent=2))

    print("\nOr test individually:")
    for level, ids in all_ids.items():
        if ids:
            print(f"\n{level.upper()} RISK IDs:")
            print(f"  {json.dumps(ids)}")

    return all_ids


if __name__ == "__main__":
    get_patients_by_risk_level()