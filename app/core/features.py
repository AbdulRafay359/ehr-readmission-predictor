import pandas as pd
import numpy as np

# Define feature order globally to prevent scoping issues
FEATURE_ORDER = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age", "gender_Male", "gender_Female", "race_Caucasian", "race_AfricanAmerican",
    "race_Other", "admission_emergency", "admission_urgent", "admission_elective",
    "discharge_expired", "discharge_home", "discharge_other", "medication_count",
    "diabetes_med", "change", "diabetes_diagnosis_count"
]

# Standard UCI Diabetes dataset medication columns
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone"
]

AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95
}

def extract_features(record: dict) -> list:
    """
    Convert a patient record (dict) into a feature vector (list)
    matching the training feature columns.
    """
    features = {}
    
    # Numeric features with safety defaults
    features["time_in_hospital"] = float(record.get("time_in_hospital", 0))
    features["num_lab_procedures"] = float(record.get("num_lab_procedures", 0))
    features["num_procedures"] = float(record.get("num_procedures", 0))
    features["num_medications"] = float(record.get("num_medications", 0))
    features["number_outpatient"] = float(record.get("number_outpatient", 0))
    features["number_emergency"] = float(record.get("number_emergency", 0))
    features["number_inpatient"] = float(record.get("number_inpatient", 0))
    features["number_diagnoses"] = float(record.get("number_diagnoses", 0))
    
    # Age mapping
    age_str = record.get("age")
    features["age"] = AGE_MAP.get(age_str, 50)
    
    # Gender (one-hot)
    gender = record.get("gender")
    features["gender_Male"] = 1 if gender == "Male" else 0
    features["gender_Female"] = 1 if gender == "Female" else 0
    
    # Race (one-hot)
    race = record.get("race")
    features["race_Caucasian"] = 1 if race == "Caucasian" else 0
    features["race_AfricanAmerican"] = 1 if race == "AfricanAmerican" else 0
    features["race_Other"] = 1 if race and race not in ["Caucasian", "AfricanAmerican"] else 0
    
    # Admission type (one-hot)
    adm_type = record.get("admission_type_id")
    features["admission_emergency"] = 1 if adm_type == 1 else 0
    features["admission_urgent"] = 1 if adm_type == 2 else 0
    features["admission_elective"] = 1 if adm_type == 3 else 0
    
    # Discharge disposition
    disch = record.get("discharge_disposition_id")
    features["discharge_expired"] = 1 if disch == 11 else 0
    features["discharge_home"] = 1 if disch in [1, 6] else 0
    features["discharge_other"] = 1 if disch is not None and disch not in [11, 1, 6] else 0
    
    # Active diabetes medication count
    med_count = sum(1 for m in MED_COLS if record.get(m) in ["Steady", "Up", "Down"])
    features["medication_count"] = med_count
    
    # Flags
    features["diabetes_med"] = 1 if record.get("diabetesMed") == "Yes" else 0
    features["change"] = 1 if record.get("change") == "Ch" else 0
    
    # Diagnostic codes check (ICD-9 codes for Diabetes start with '250')
    diag_codes = [record.get("diag_1"), record.get("diag_2"), record.get("diag_3")]
    diabetes_code_count = sum(
        1 for code in diag_codes 
        if code is not None and str(code).startswith("250")
    )
    features["diabetes_diagnosis_count"] = diabetes_code_count
    
    # Return vector strictly in defined order
    return [features[col] for col in FEATURE_ORDER]

def get_feature_names() -> list:
    return FEATURE_ORDER