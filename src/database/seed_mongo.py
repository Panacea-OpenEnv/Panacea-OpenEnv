"""
MongoDB Seed Script
Populates all 9 collections with realistic hospital data.
Run once: python -m src.database.seed_mongo
"""

import sys
from datetime import datetime, timedelta
from src.database.mongo_client import (
    get_sync_db, create_indexes, ping_db,
    PATIENTS, VITALS, COMORBIDITIES, PROTOCOLS,
    RESOURCES, CLAIMS, PATIENT_CONSULTATIONS,
    SPECIALIST_REPORTS, MEDICAL_SUMMARIES,
)

def seed_patients(db):
    data = [
        {"patient_id": "P1001", "name": "Arjun Mehta",      "age": 58, "gender": "M", "blood_group": "O+", "phone": "9876543210"},
        {"patient_id": "P1002", "name": "Priya Sharma",     "age": 34, "gender": "F", "blood_group": "A+", "phone": "9123456780"},
        {"patient_id": "P1003", "name": "Ravi Kumar",       "age": 72, "gender": "M", "blood_group": "B+", "phone": "9988776655"},
        {"patient_id": "P1004", "name": "Sunita Patel",     "age": 45, "gender": "F", "blood_group": "AB+","phone": "9871234560"},
        {"patient_id": "P1005", "name": "Vikram Singh",     "age": 61, "gender": "M", "blood_group": "O-", "phone": "9765432100"},
        {"patient_id": "P1006", "name": "Anjali Desai",     "age": 29, "gender": "F", "blood_group": "A-", "phone": "9654321009"},
        {"patient_id": "P1007", "name": "Mohammed Raza",    "age": 55, "gender": "M", "blood_group": "B-", "phone": "9543210098"},
        {"patient_id": "P1008", "name": "Kavitha Nair",     "age": 42, "gender": "F", "blood_group": "O+", "phone": "9432100987"},
        {"patient_id": "P1009", "name": "Deepak Joshi",     "age": 67, "gender": "M", "blood_group": "A+", "phone": "9321009876"},
        {"patient_id": "P1010", "name": "Lakshmi Reddy",    "age": 38, "gender": "F", "blood_group": "B+", "phone": "9210098765"},
        {"patient_id": "P1011", "name": "Suresh Iyer",      "age": 50, "gender": "M", "blood_group": "AB-","phone": "9109876540"},
        {"patient_id": "P1012", "name": "Meena Pillai",     "age": 63, "gender": "F", "blood_group": "O+", "phone": "9098765430"},
        {"patient_id": "P1013", "name": "Arun Bose",        "age": 44, "gender": "M", "blood_group": "A+", "phone": "8987654320"},
        {"patient_id": "P1014", "name": "Divya Menon",      "age": 31, "gender": "F", "blood_group": "B+", "phone": "8876543210"},
        {"patient_id": "P1015", "name": "Rajesh Verma",     "age": 76, "gender": "M", "blood_group": "O+", "phone": "8765432100"},
        {"patient_id": "P1016", "name": "Nisha Gupta",      "age": 27, "gender": "F", "blood_group": "A+", "phone": "8654321009"},
        {"patient_id": "P1017", "name": "Harish Choudhary", "age": 53, "gender": "M", "blood_group": "O-", "phone": "8543210098"},
        {"patient_id": "P1018", "name": "Rekha Saxena",     "age": 48, "gender": "F", "blood_group": "B+", "phone": "8432100987"},
        {"patient_id": "P1019", "name": "Gopal Das",        "age": 70, "gender": "M", "blood_group": "A+", "phone": "8321009876"},
        {"patient_id": "P1020", "name": "Sarita Rao",       "age": 36, "gender": "F", "blood_group": "AB+","phone": "8210098765"},
    ]
    db[PATIENTS].delete_many({})
    db[PATIENTS].insert_many(data)
    print(f"  Seeded {len(data)} patients")

def seed_vitals(db):
    base = datetime.utcnow()
    data = [
        {"patient_id": "P1001", "heart_rate": 102, "bp_systolic": 158, "bp_diastolic": 96,  "temperature": 37.2, "o2_saturation": 94, "respiratory_rate": 20, "recorded_at": base - timedelta(hours=1)},
        {"patient_id": "P1002", "heart_rate": 88,  "bp_systolic": 130, "bp_diastolic": 82,  "temperature": 38.5, "o2_saturation": 97, "respiratory_rate": 18, "recorded_at": base - timedelta(hours=2)},
        {"patient_id": "P1003", "heart_rate": 78,  "bp_systolic": 170, "bp_diastolic": 100, "temperature": 36.8, "o2_saturation": 91, "respiratory_rate": 22, "recorded_at": base - timedelta(hours=3)},
        {"patient_id": "P1004", "heart_rate": 95,  "bp_systolic": 145, "bp_diastolic": 90,  "temperature": 37.0, "o2_saturation": 96, "respiratory_rate": 19, "recorded_at": base - timedelta(hours=1)},
        {"patient_id": "P1005", "heart_rate": 110, "bp_systolic": 160, "bp_diastolic": 98,  "temperature": 37.5, "o2_saturation": 93, "respiratory_rate": 24, "recorded_at": base - timedelta(hours=2)},
        {"patient_id": "P1006", "heart_rate": 72,  "bp_systolic": 118, "bp_diastolic": 76,  "temperature": 36.6, "o2_saturation": 99, "respiratory_rate": 16, "recorded_at": base - timedelta(hours=4)},
        {"patient_id": "P1007", "heart_rate": 85,  "bp_systolic": 140, "bp_diastolic": 88,  "temperature": 37.8, "o2_saturation": 95, "respiratory_rate": 20, "recorded_at": base - timedelta(hours=1)},
        {"patient_id": "P1008", "heart_rate": 92,  "bp_systolic": 135, "bp_diastolic": 85,  "temperature": 39.1, "o2_saturation": 96, "respiratory_rate": 21, "recorded_at": base - timedelta(hours=3)},
        {"patient_id": "P1009", "heart_rate": 68,  "bp_systolic": 155, "bp_diastolic": 94,  "temperature": 36.9, "o2_saturation": 90, "respiratory_rate": 23, "recorded_at": base - timedelta(hours=2)},
        {"patient_id": "P1010", "heart_rate": 80,  "bp_systolic": 122, "bp_diastolic": 78,  "temperature": 37.1, "o2_saturation": 98, "respiratory_rate": 17, "recorded_at": base - timedelta(hours=5)},
        {"patient_id": "P1011", "heart_rate": 76,  "bp_systolic": 148, "bp_diastolic": 92,  "temperature": 37.3, "o2_saturation": 94, "respiratory_rate": 19, "recorded_at": base - timedelta(hours=1)},
        {"patient_id": "P1012", "heart_rate": 88,  "bp_systolic": 162, "bp_diastolic": 98,  "temperature": 36.7, "o2_saturation": 92, "respiratory_rate": 22, "recorded_at": base - timedelta(hours=2)},
        {"patient_id": "P1013", "heart_rate": 96,  "bp_systolic": 138, "bp_diastolic": 86,  "temperature": 38.2, "o2_saturation": 95, "respiratory_rate": 20, "recorded_at": base - timedelta(hours=3)},
        {"patient_id": "P1014", "heart_rate": 74,  "bp_systolic": 115, "bp_diastolic": 72,  "temperature": 36.5, "o2_saturation": 99, "respiratory_rate": 15, "recorded_at": base - timedelta(hours=6)},
        {"patient_id": "P1015", "heart_rate": 64,  "bp_systolic": 175, "bp_diastolic": 102, "temperature": 36.8, "o2_saturation": 89, "respiratory_rate": 25, "recorded_at": base - timedelta(hours=1)},
        {"patient_id": "P1016", "heart_rate": 70,  "bp_systolic": 112, "bp_diastolic": 70,  "temperature": 36.4, "o2_saturation": 100,"respiratory_rate": 14, "recorded_at": base - timedelta(hours=4)},
        {"patient_id": "P1017", "heart_rate": 82,  "bp_systolic": 144, "bp_diastolic": 90,  "temperature": 37.6, "o2_saturation": 93, "respiratory_rate": 21, "recorded_at": base - timedelta(hours=2)},
        {"patient_id": "P1018", "heart_rate": 90,  "bp_systolic": 136, "bp_diastolic": 84,  "temperature": 38.8, "o2_saturation": 95, "respiratory_rate": 20, "recorded_at": base - timedelta(hours=3)},
        {"patient_id": "P1019", "heart_rate": 66,  "bp_systolic": 168, "bp_diastolic": 100, "temperature": 36.9, "o2_saturation": 88, "respiratory_rate": 26, "recorded_at": base - timedelta(hours=1)},
        {"patient_id": "P1020", "heart_rate": 78,  "bp_systolic": 120, "bp_diastolic": 76,  "temperature": 37.0, "o2_saturation": 98, "respiratory_rate": 16, "recorded_at": base - timedelta(hours=5)},
    ]
    db[VITALS].delete_many({})
    db[VITALS].insert_many(data)
    print(f"  Seeded {len(data)} vitals records")

def seed_comorbidities(db):
    data = [
        {"patient_id": "P1001", "condition": "hypertension",         "is_critical": False, "since_year": 2018},
        {"patient_id": "P1001", "condition": "type2_diabetes",        "is_critical": False, "since_year": 2020},
        {"patient_id": "P1002", "condition": "asthma",                "is_critical": True,  "since_year": 2015},
        {"patient_id": "P1003", "condition": "hemophilia",            "is_critical": True,  "since_year": 2000},
        {"patient_id": "P1003", "condition": "chronic_kidney_disease","is_critical": True,  "since_year": 2019},
        {"patient_id": "P1004", "condition": "hypothyroidism",        "is_critical": False, "since_year": 2017},
        {"patient_id": "P1005", "condition": "COPD",                  "is_critical": True,  "since_year": 2016},
        {"patient_id": "P1005", "condition": "hypertension",          "is_critical": False, "since_year": 2014},
        {"patient_id": "P1006", "condition": "migraine",              "is_critical": False, "since_year": 2019},
        {"patient_id": "P1007", "condition": "type2_diabetes",        "is_critical": False, "since_year": 2018},
        {"patient_id": "P1007", "condition": "hypertension",          "is_critical": False, "since_year": 2016},
        {"patient_id": "P1008", "condition": "rheumatoid_arthritis",  "is_critical": False, "since_year": 2020},
        {"patient_id": "P1009", "condition": "heart_failure",         "is_critical": True,  "since_year": 2021},
        {"patient_id": "P1009", "condition": "atrial_fibrillation",   "is_critical": True,  "since_year": 2022},
        {"patient_id": "P1010", "condition": "lupus",                 "is_critical": True,  "since_year": 2017},
        {"patient_id": "P1011", "condition": "hypertension",          "is_critical": False, "since_year": 2019},
        {"patient_id": "P1012", "condition": "osteoporosis",          "is_critical": False, "since_year": 2020},
        {"patient_id": "P1013", "condition": "epilepsy",              "is_critical": True,  "since_year": 2010},
        {"patient_id": "P1014", "condition": "PCOS",                  "is_critical": False, "since_year": 2018},
        {"patient_id": "P1015", "condition": "Parkinson_disease",     "is_critical": True,  "since_year": 2019},
        {"patient_id": "P1015", "condition": "hypertension",          "is_critical": False, "since_year": 2012},
        {"patient_id": "P1016", "condition": "anemia",                "is_critical": False, "since_year": 2022},
        {"patient_id": "P1017", "condition": "type2_diabetes",        "is_critical": False, "since_year": 2017},
        {"patient_id": "P1018", "condition": "psoriasis",             "is_critical": False, "since_year": 2016},
        {"patient_id": "P1019", "condition": "prostate_cancer",       "is_critical": True,  "since_year": 2023},
        {"patient_id": "P1020", "condition": "endometriosis",         "is_critical": False, "since_year": 2021},
    ]
    db[COMORBIDITIES].delete_many({})
    db[COMORBIDITIES].insert_many(data)
    print(f"  Seeded {len(data)} comorbidity records")

def seed_protocols(db):
    data = [
        {"specialty": "Cardiology",        "protocol": "ACLS",               "steps": ["12-lead ECG", "Troponin assay", "Aspirin 325mg stat", "PCI within 90 min"]},
        {"specialty": "Neurology",         "protocol": "Stroke Protocol",    "steps": ["CT head stat", "NIHSS score", "tPA within 4.5h if eligible", "Neurology ICU admit"]},
        {"specialty": "Neurosurgery",      "protocol": "Craniotomy Protocol","steps": ["Pre-op CT/MRI", "Anesthesia consult", "OR booking", "ICP monitoring post-op"]},
        {"specialty": "Orthopedics",       "protocol": "Fracture Reduction", "steps": ["X-ray bilateral", "Pain management", "Reduction under sedation", "Cast/fixation"]},
        {"specialty": "Pediatrics",        "protocol": "Pediatric ACLS",     "steps": ["Weight-based dosing", "IV access", "Fluid resuscitation", "Pediatric ICU if needed"]},
        {"specialty": "Gynecology",        "protocol": "Laparoscopy",        "steps": ["Pre-op bloods", "Anesthesia consult", "Laparoscopic entry", "Post-op monitoring"]},
        {"specialty": "Obstetrics",        "protocol": "C-Section Protocol", "steps": ["Spinal anesthesia", "Surgical prep", "Uterine incision", "Neonatal team standby"]},
        {"specialty": "Dermatology",       "protocol": "Skin Biopsy",        "steps": ["Local anesthesia", "Punch/excision biopsy", "Histopathology send", "Wound care"]},
        {"specialty": "Ophthalmology",     "protocol": "Retinal Detachment", "steps": ["Dilated fundus exam", "OCT scan", "Emergency vitreoretinal surgery", "Post-op posturing"]},
        {"specialty": "Otolaryngology",    "protocol": "Tonsillectomy",      "steps": ["Pre-op assessment", "General anesthesia", "Tonsil removal", "Post-op airway monitoring"]},
        {"specialty": "Gastroenterology",  "protocol": "Upper GI Endoscopy", "steps": ["NPO 6h", "Sedation", "Endoscope insertion", "Biopsy if needed", "Recovery"]},
        {"specialty": "Pulmonology",       "protocol": "Mechanical Ventilation", "steps": ["Intubation", "Ventilator settings", "ABG monitoring", "Weaning protocol"]},
        {"specialty": "Nephrology",        "protocol": "Hemodialysis",       "steps": ["AV fistula/catheter access", "Heparin anticoagulation", "4h session", "Post-dialysis monitoring"]},
        {"specialty": "Urology",           "protocol": "Lithotripsy",        "steps": ["Pre-op KUB X-ray", "Anesthesia", "ESWL session", "Post-op urine straining"]},
        {"specialty": "Endocrinology",     "protocol": "Insulin Protocol",   "steps": ["Glucose monitoring", "Basal-bolus insulin", "HbA1c target <7%", "Diabetologist review"]},
        {"specialty": "Oncology",          "protocol": "Chemotherapy",       "steps": ["Staging workup", "MDT discussion", "Port insertion", "Chemo cycles", "CBC monitoring"]},
        {"specialty": "Hematology",        "protocol": "Transfusion Protocol","steps": ["Blood type & crossmatch", "IV access", "Transfuse over 2-4h", "Post-transfusion Hb check"]},
        {"specialty": "Rheumatology",      "protocol": "DMARDs",             "steps": ["Baseline LFT/CBC", "Methotrexate 7.5mg weekly", "Folic acid", "3-monthly monitoring"]},
        {"specialty": "Psychiatry",        "protocol": "Crisis Intervention", "steps": ["Safety assessment", "De-escalation", "Medication review", "Inpatient admission if needed"]},
        {"specialty": "Radiology",         "protocol": "Contrast CT",        "steps": ["Renal function check", "IV contrast admin", "Scan acquisition", "Report within 1h"]},
        {"specialty": "Anesthesiology",    "protocol": "General Anesthesia", "steps": ["Pre-op assessment", "Fasting check", "Induction", "Intubation", "Maintenance", "Emergence"]},
        {"specialty": "General Medicine",  "protocol": "Sepsis Bundle",      "steps": ["Blood cultures x2", "Lactate", "IV fluids 30ml/kg", "Broad-spectrum antibiotics within 1h"]},
        {"specialty": "Pathology",         "protocol": "Biopsy Processing",  "steps": ["Formalin fixation", "Tissue processing", "Paraffin embedding", "H&E staining", "Report"]},
        {"specialty": "Plastic Surgery",   "protocol": "Burn Management",    "steps": ["Fluid resuscitation (Parkland)", "Wound debridement", "Skin grafting", "Infection monitoring"]},
        {"specialty": "Vascular Surgery",  "protocol": "Aortic Repair",      "steps": ["CT angiography", "Anaesthesia consult", "EVAR or open repair", "ICU post-op"]},
        {"specialty": "Infectious Disease","protocol": "Sepsis Bundle",      "steps": ["Source identification", "Culture-directed antibiotics", "Source control", "De-escalation"]},
    ]
    db[PROTOCOLS].delete_many({})
    db[PROTOCOLS].insert_many(data)
    print(f"  Seeded {len(data)} protocols")

def seed_resources(db):
    data = [
        {"resource_id": "R001", "resource_type": "bed",       "name": "Cardiac ICU Bed",          "available": 4,  "total": 6},
        {"resource_id": "R002", "resource_type": "bed",       "name": "General Ward Bed",          "available": 20, "total": 30},
        {"resource_id": "R003", "resource_type": "bed",       "name": "Neurology ICU Bed",         "available": 2,  "total": 4},
        {"resource_id": "R004", "resource_type": "bed",       "name": "Pediatric Ward Bed",        "available": 8,  "total": 12},
        {"resource_id": "R005", "resource_type": "bed",       "name": "Psychiatric Ward Bed",      "available": 5,  "total": 8},
        {"resource_id": "R006", "resource_type": "equipment", "name": "Mechanical Ventilator",     "available": 3,  "total": 8},
        {"resource_id": "R007", "resource_type": "equipment", "name": "ECG Machine",               "available": 5,  "total": 5},
        {"resource_id": "R008", "resource_type": "equipment", "name": "Echocardiogram Unit",       "available": 2,  "total": 3},
        {"resource_id": "R009", "resource_type": "equipment", "name": "MRI Machine",               "available": 1,  "total": 2},
        {"resource_id": "R010", "resource_type": "equipment", "name": "CT Scanner",                "available": 2,  "total": 2},
        {"resource_id": "R011", "resource_type": "equipment", "name": "Dialysis Machine",          "available": 2,  "total": 4},
        {"resource_id": "R012", "resource_type": "equipment", "name": "Defibrillator",             "available": 6,  "total": 6},
        {"resource_id": "R013", "resource_type": "or_suite",  "name": "Cardiac OR Suite",          "available": 1,  "total": 2},
        {"resource_id": "R014", "resource_type": "or_suite",  "name": "General OR Suite",          "available": 2,  "total": 4},
        {"resource_id": "R015", "resource_type": "or_suite",  "name": "Neurosurgery OR Suite",     "available": 1,  "total": 1},
        {"resource_id": "R016", "resource_type": "drug",      "name": "Aspirin 325mg",             "available": 500,"total": 500},
        {"resource_id": "R017", "resource_type": "drug",      "name": "Nitroglycerin 0.4mg",       "available": 200,"total": 200},
        {"resource_id": "R018", "resource_type": "drug",      "name": "Metoprolol 25mg",           "available": 300,"total": 300},
        {"resource_id": "R019", "resource_type": "drug",      "name": "Metformin 500mg",           "available": 400,"total": 400},
        {"resource_id": "R020", "resource_type": "drug",      "name": "Insulin (Basal)",           "available": 100,"total": 100},
        {"resource_id": "R021", "resource_type": "drug",      "name": "IV Broad-spectrum Antibiotics","available": 150,"total": 150},
        {"resource_id": "R022", "resource_type": "drug",      "name": "tPA (Alteplase)",           "available": 10, "total": 10},
        {"resource_id": "R023", "resource_type": "drug",      "name": "Coagulation Factor VIII",   "available": 20, "total": 20},
        {"resource_id": "R024", "resource_type": "lab",       "name": "Troponin Assay",            "available": 50, "total": 50},
        {"resource_id": "R025", "resource_type": "lab",       "name": "Blood Culture Kit",         "available": 40, "total": 40},
    ]
    db[RESOURCES].delete_many({})
    db[RESOURCES].insert_many(data)
    print(f"  Seeded {len(data)} resources")

def seed_claims(db):
    data = [
        {"claim_id": "C001", "patient_id": "P1001", "department": "Cardiology",    "resource": "Cardiac ICU Bed",    "claimed_amount": 2.0, "actual_amount": 2.0, "status": "pending",  "deceptive": False},
        {"claim_id": "C002", "patient_id": "P1003", "department": "Hematology",    "resource": "Coagulation Factor VIII","claimed_amount": 10.0,"actual_amount": 4.0,"status": "pending","deceptive": True},
        {"claim_id": "C003", "patient_id": "P1005", "department": "Pulmonology",   "resource": "Mechanical Ventilator",  "claimed_amount": 1.0,"actual_amount": 1.0, "status": "pending",  "deceptive": False},
        {"claim_id": "C004", "patient_id": "P9999", "department": "Neurology",     "resource": "Neurology ICU Bed",  "claimed_amount": 1.0, "actual_amount": 0.0, "status": "pending",  "deceptive": True},
        {"claim_id": "C005", "patient_id": "P1009", "department": "Cardiology",    "resource": "Cardiac ICU Bed",    "claimed_amount": 3.0, "actual_amount": 1.0, "status": "pending",  "deceptive": True},
        {"claim_id": "C006", "patient_id": "P1002", "department": "Pulmonology",   "resource": "Mechanical Ventilator",  "claimed_amount": 1.0,"actual_amount": 1.0, "status": "approved", "deceptive": False},
        {"claim_id": "C007", "patient_id": "P1007", "department": "Endocrinology", "resource": "Insulin (Basal)",    "claimed_amount": 2.0, "actual_amount": 2.0, "status": "approved", "deceptive": False},
        {"claim_id": "C008", "patient_id": "P1013", "department": "Neurology",     "resource": "MRI Machine",        "claimed_amount": 1.0, "actual_amount": 1.0, "status": "approved", "deceptive": False},
        {"claim_id": "C009", "patient_id": "P1015", "department": "Neurology",     "resource": "Neurology ICU Bed",  "claimed_amount": 2.0, "actual_amount": 1.0, "status": "pending",  "deceptive": True},
        {"claim_id": "C010", "patient_id": "P1019", "department": "Oncology",      "resource": "Cardiac ICU Bed",    "claimed_amount": 1.0, "actual_amount": 0.0, "status": "pending",  "deceptive": True},
    ]
    db[CLAIMS].delete_many({})
    db[CLAIMS].insert_many(data)
    print(f"  Seeded {len(data)} claims")

def run_seed():
    if not ping_db():
        print("MongoDB connection failed. Check MONGODB_URI in .env")
        sys.exit(1)

    print("\nSeeding MongoDB — database: panacea")
    print("-" * 40)
    create_indexes()
    db = get_sync_db()
    seed_patients(db)
    seed_vitals(db)
    seed_comorbidities(db)
    seed_protocols(db)
    seed_resources(db)
    seed_claims(db)
    print("-" * 40)
    print("Seed complete. All collections populated.")

if __name__ == "__main__":
    run_seed()
