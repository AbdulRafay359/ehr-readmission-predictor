import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "ehr_readmission_db")
DATA_DIR = "data/raw/"

def connect_db():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db

def load_mappings(db):
    """Load IDS_mapping.csv into separate collections."""
    df = pd.read_csv(os.path.join(DATA_DIR, "IDS_mapping.csv"))
    sections = {}
    current_section = None
    current_data = []

    for idx, row in df.iterrows():
        if pd.notna(row.iloc[0]) and isinstance(row.iloc[0], str):
            lower = row.iloc[0].lower()
            if "admission_type_id" in lower:
                current_section = "admission_type"
                current_data = []
                continue
            elif "discharge_disposition_id" in lower:
                current_section = "discharge_disposition"
                current_data = []
                continue
            elif "admission_source_id" in lower:
                current_section = "admission_source"
                current_data = []
                continue

        if pd.isna(row.iloc[0]) and pd.isna(row.iloc[1]):
            continue

        if current_section and pd.notna(row.iloc[0]):
            current_data.append({
                "code": int(row.iloc[0]),
                "description": str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
            })

        # Save when next header or end
        if current_section and current_data:
            if idx + 1 < len(df):
                next_row = df.iloc[idx + 1]
                if pd.notna(next_row.iloc[0]) and isinstance(next_row.iloc[0], str):
                    if any(k in str(next_row.iloc[0]).lower() for k in ["admission_type", "discharge_disposition", "admission_source"]):
                        sections[current_section] = current_data
                        current_data = []
            elif idx == len(df) - 1:
                sections[current_section] = current_data

    for section_name, data in sections.items():
        coll = db[f"mappings_{section_name}"]
        coll.delete_many({})
        result = coll.insert_many(data)
        print(f"Loaded {len(result.inserted_ids)} records into {section_name} mapping")
    return sections

def load_patient_data(db):
    """Load diabetic_data.csv with proper types and cleaning."""
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetic_data.csv"), dtype={
        "encounter_id": "int64",
        "patient_nbr": "int64",
        "race": "object",
        "gender": "object",
        "age": "object",
        "weight": "object",
        "admission_type_id": "int64",
        "discharge_disposition_id": "int64",
        "admission_source_id": "int64",
        "time_in_hospital": "int64",
        "payer_code": "object",
        "medical_specialty": "object",
        "num_lab_procedures": "int64",
        "num_procedures": "int64",
        "num_medications": "int64",
        "number_outpatient": "int64",
        "number_emergency": "int64",
        "number_inpatient": "int64",
        "diag_1": "object",
        "diag_2": "object",
        "diag_3": "object",
        "number_diagnoses": "int64",
        "max_glu_serum": "object",
        "A1Cresult": "object",
        "metformin": "object",
        "repaglinide": "object",
        "nateglinide": "object",
        "chlorpropamide": "object",
        "glimepiride": "object",
        "acetohexamide": "object",
        "glipizide": "object",
        "glyburide": "object",
        "tolbutamide": "object",
        "pioglitazone": "object",
        "rosiglitazone": "object",
        "acarbose": "object",
        "miglitol": "object",
        "troglitazone": "object",
        "tolazamide": "object",
        "examide": "object",
        "citoglipton": "object",
        "insulin": "object",
        "glyburide-metformin": "object",
        "glipizide-metformin": "object",
        "glimepiride-pioglitazone": "object",
        "metformin-rosiglitazone": "object",
        "metformin-pioglitazone": "object",
        "change": "object",
        "diabetesMed": "object",
        "readmitted": "object"
    })
    df = df.replace("?", None)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    records = df.to_dict("records")
    cleaned = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if k is None or k == "":
                continue
            if pd.isna(v):
                clean[k] = None
            elif v == "?":
                clean[k] = None
            elif isinstance(v, (int, float)):
                clean[k] = v
            else:
                clean[k] = str(v)
        clean["_metadata"] = {
            "import_date": datetime.utcnow().isoformat(),
            "source": "UCI Diabetes"
        }
        cleaned.append(clean)

    coll = db["diabetes_patients"]
    coll.delete_many({})
    batch_size = 10000
    total = 0
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        res = coll.insert_many(batch)
        total += len(res.inserted_ids)
    print(f"Inserted {total} patient records")

    # Indexes
    coll.create_index("encounter_id", unique=True)
    coll.create_index("patient_nbr")
    coll.create_index("readmitted")
    coll.create_index("admission_type_id")
    coll.create_index("discharge_disposition_id")
    return cleaned

def create_combined_view(db):
    """Join patient data with mapping descriptions and derive target."""
    patients = list(db.diabetes_patients.find({}))
    maps = {
        "admission_type": {item["code"]: item["description"] for item in db.mappings_admission_type.find({})},
        "discharge": {item["code"]: item["description"] for item in db.mappings_discharge_disposition.find({})},
        "admission_source": {item["code"]: item["description"] for item in db.mappings_admission_source.find({})}
    }

    combined = []
    for pat in patients:
        p = pat.copy()
        p["admission_type_desc"] = maps["admission_type"].get(p.get("admission_type_id"), "Unknown")
        p["discharge_disposition_desc"] = maps["discharge"].get(p.get("discharge_disposition_id"), "Unknown")
        p["admission_source_desc"] = maps["admission_source"].get(p.get("admission_source_id"), "Unknown")
        # target
        p["readmitted_30d"] = 1 if p.get("readmitted") == "<30" else 0
        combined.append(p)

    coll = db["diabetes_patients_combined"]
    coll.delete_many({})
    res = coll.insert_many(combined)
    print(f"Created combined view with {len(res.inserted_ids)} records")
    return combined

if __name__ == "__main__":
    db = connect_db()
    load_mappings(db)
    load_patient_data(db)
    create_combined_view(db)