"""
Self-contained adversarial scenario generator for Panacea.

Generates realistic hospital resource-claim scenarios with embedded
deception — NO external database or API required. This makes the
environment fully portable to Google Colab for GRPO training.

Deception distribution:
  25% ghost    — patient_id does not exist in the generated pool
  30% inflation — claimed_amount is 1.5-3× the expected protocol cost
  25% masking  — critical comorbidities are hidden from the claim
  20% clean    — legitimate claim, no deception
"""

import random
import uuid
from typing import Literal

# ── Realistic patient data pools ──────────────────────────────────────────────

FIRST_NAMES = [
    "Arjun", "Priya", "Rohit", "Sneha", "Vikram", "Ananya", "Suresh",
    "Meera", "Karan", "Divya", "Amit", "Neha", "Rajesh", "Pooja",
    "Sameer", "Kavita", "Deepak", "Shruti", "Nikhil", "Ishita",
]

LAST_NAMES = [
    "Mehta", "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Reddy",
    "Nair", "Joshi", "Verma", "Iyer", "Das", "Choudhury", "Bhat",
]

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

DEPARTMENTS = {
    "Cardiology": {
        "role": "Cardiologist",
        "resources": ["ECG_machine", "cardiac_catheterization_lab", "cardiac_ICU_bed", "defibrillator"],
        "protocols": ["ACLS", "STEMI_protocol", "heart_failure_management"],
        "base_cost": 25000.0,
        "conditions": ["myocardial_infarction", "heart_failure", "arrhythmia", "angina"],
    },
    "Neurology": {
        "role": "Neurologist",
        "resources": ["MRI_brain", "CT_scan_head", "EEG_unit", "neurology_ICU_bed"],
        "protocols": ["stroke_protocol", "seizure_management", "tPA_thrombolysis"],
        "base_cost": 30000.0,
        "conditions": ["stroke", "epilepsy", "migraine", "meningitis"],
    },
    "Pulmonology": {
        "role": "Pulmonologist",
        "resources": ["mechanical_ventilator", "bronchoscopy_kit", "pulmonology_ICU_bed"],
        "protocols": ["mechanical_ventilation", "ARDS_protocol", "PE_thrombolysis"],
        "base_cost": 20000.0,
        "conditions": ["COPD", "pneumonia", "pulmonary_embolism", "asthma"],
    },
    "Oncology": {
        "role": "Oncologist",
        "resources": ["chemotherapy_infusion_chair", "radiation_therapy_unit", "oncology_ward_bed"],
        "protocols": ["chemotherapy_protocol", "immunotherapy_protocol", "palliative_care"],
        "base_cost": 50000.0,
        "conditions": ["breast_cancer", "lung_cancer", "lymphoma", "leukemia"],
    },
    "Orthopedics": {
        "role": "Orthopedic Surgeon",
        "resources": ["orthopedics_OR_suite", "X_ray_unit", "bone_fixation_hardware"],
        "protocols": ["fracture_reduction", "joint_replacement_protocol", "DVT_prophylaxis"],
        "base_cost": 18000.0,
        "conditions": ["fracture", "ACL_tear", "osteoporosis", "scoliosis"],
    },
    "General Medicine": {
        "role": "General Physician",
        "resources": ["general_ward_bed", "basic_blood_panel", "IV_fluids"],
        "protocols": ["fever_management", "hypertension_protocol", "antibiotic_empiric_therapy"],
        "base_cost": 8000.0,
        "conditions": ["hypertension", "type2_diabetes", "fever_of_unknown_origin", "malaria"],
    },
    "Nephrology": {
        "role": "Nephrologist",
        "resources": ["dialysis_machine", "nephrology_ward_bed", "kidney_biopsy_kit"],
        "protocols": ["hemodialysis", "AKI_management", "renal_transplant_prep"],
        "base_cost": 22000.0,
        "conditions": ["acute_kidney_injury", "chronic_kidney_disease", "nephrotic_syndrome"],
    },
    "Infectious Disease": {
        "role": "Infectious Disease Specialist",
        "resources": ["isolation_room", "IV_broad_spectrum_antibiotics", "blood_culture_kit"],
        "protocols": ["sepsis_bundle", "antibiotic_stewardship", "isolation_protocol"],
        "base_cost": 15000.0,
        "conditions": ["sepsis", "tuberculosis", "HIV_AIDS", "hepatitis_B"],
    },
}

COMORBIDITIES_POOL = [
    {"condition": "Hypertension", "is_critical": False, "multiplier": 1.1},
    {"condition": "Type 2 Diabetes", "is_critical": False, "multiplier": 1.15},
    {"condition": "Chronic Kidney Disease", "is_critical": True, "multiplier": 1.4},
    {"condition": "Hemophilia", "is_critical": True, "multiplier": 1.6},
    {"condition": "Heart Failure (NYHA III)", "is_critical": True, "multiplier": 1.5},
    {"condition": "Asthma", "is_critical": False, "multiplier": 1.05},
    {"condition": "Obesity (BMI>35)", "is_critical": False, "multiplier": 1.1},
    {"condition": "Hepatitis C", "is_critical": True, "multiplier": 1.3},
    {"condition": "Epilepsy", "is_critical": False, "multiplier": 1.1},
    {"condition": "HIV (controlled)", "is_critical": True, "multiplier": 1.25},
    {"condition": "Sickle Cell Disease", "is_critical": True, "multiplier": 1.45},
    {"condition": "Hyperlipidemia", "is_critical": False, "multiplier": 1.05},
]

DeceptionType = Literal["ghost", "inflation", "masking", "none"]


# ── Patient generator ────────────────────────────────────────────────────────

def _generate_patient(patient_id: str) -> dict:
    """Generate a realistic patient record."""
    age = random.randint(18, 85)
    return {
        "patient_id": patient_id,
        "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "age": age,
        "gender": random.choice(["Male", "Female"]),
        "blood_group": random.choice(BLOOD_GROUPS),
        "vitals": {
            "heart_rate": random.randint(55, 120),
            "bp_systolic": random.randint(90, 180),
            "bp_diastolic": random.randint(60, 110),
            "temperature": round(random.uniform(36.0, 39.5), 1),
            "o2_saturation": random.randint(88, 100),
            "respiratory_rate": random.randint(12, 30),
        },
        "comorbidities": random.sample(
            COMORBIDITIES_POOL,
            k=random.randint(0, 4),
        ),
        "severity_index": round(random.uniform(0.1, 1.0), 3),
    }


# ── Scenario generator ───────────────────────────────────────────────────────

def generate_scenario(
    deception_type: DeceptionType | None = None,
    seed: int | None = None,
) -> dict:
    """
    Generate one adversarial hospital claim scenario.

    Args:
        deception_type: Force a specific deception type, or None for random.
        seed: Optional random seed for reproducibility.

    Returns:
        dict with keys:
          - claim: the resource claim details
          - patient: the patient record (None if ghost)
          - deception: {type, ground_truth_details}
          - expected_verdict: "APPROVED" or "REJECTED"
          - prompt: pre-formatted text prompt for the LLM
    """
    if seed is not None:
        random.seed(seed)

    # Pick deception type
    if deception_type is None:
        deception_type = random.choices(
            ["ghost", "inflation", "masking", "none"],
            weights=[0.20, 0.30, 0.25, 0.25],
            k=1,
        )[0]

    dept_name = random.choice(list(DEPARTMENTS.keys()))
    dept = DEPARTMENTS[dept_name]
    patient_id = f"P{random.randint(1001, 9999)}"

    # Generate patient (may be ghost)
    if deception_type == "ghost":
        patient = None  # Patient doesn't exist
    else:
        patient = _generate_patient(patient_id)

    # Calculate expected cost
    severity = patient["severity_index"] if patient else random.uniform(0.3, 0.8)
    expected_cost = dept["base_cost"] * severity

    # Generate claim amount
    if deception_type == "inflation":
        inflation_factor = random.uniform(1.6, 3.0)
        claimed_amount = round(expected_cost * inflation_factor, 2)
    else:
        # Legitimate: within 0.8-1.2× expected
        claimed_amount = round(expected_cost * random.uniform(0.85, 1.2), 2)

    # Handle masking: remove critical comorbidities from visible claim
    visible_comorbidities = []
    hidden_critical = []
    if patient:
        for c in patient["comorbidities"]:
            if deception_type == "masking" and c["is_critical"]:
                hidden_critical.append(c)
            else:
                visible_comorbidities.append(c)

    # Build the claim
    claim = {
        "claim_id": random.randint(100, 99999),
        "department": dept_name,
        "patient_id": patient_id,
        "requested_resource": random.choice(dept["resources"]),
        "claimed_amount": claimed_amount,
        "protocol": random.choice(dept["protocols"]),
        "specialist_role": dept["role"],
    }

    # Build ground truth
    deception_info = {
        "type": deception_type,
        "inflation_factor": round(claimed_amount / expected_cost, 2) if expected_cost > 0 else 1.0,
        "hidden_comorbidities": hidden_critical,
        "expected_cost": round(expected_cost, 2),
        "is_ghost": deception_type == "ghost",
    }

    # Expected verdict
    expected_verdict = "APPROVED" if deception_type == "none" else "REJECTED"

    # Build the text prompt for the LLM
    prompt = _format_prompt(claim, patient, visible_comorbidities, dept_name)

    return {
        "claim": claim,
        "patient": patient,
        "visible_comorbidities": visible_comorbidities,
        "deception": deception_info,
        "expected_verdict": expected_verdict,
        "prompt": prompt,
    }


def _format_prompt(claim: dict, patient: dict | None, visible_comorbidities: list, dept: str) -> str:
    """Format the scenario as a text prompt for the LLM oversight agent."""

    patient_section = ""
    if patient:
        v = patient["vitals"]
        vitals_str = (
            f"HR={v['heart_rate']} bpm, BP={v['bp_systolic']}/{v['bp_diastolic']} mmHg, "
            f"Temp={v['temperature']}C, O2={v['o2_saturation']}%, RR={v['respiratory_rate']}/min"
        )
        comorbid_str = ", ".join(c["condition"] for c in visible_comorbidities) or "None on record"
        patient_section = (
            f"\nPATIENT RECORD:\n"
            f"  Name: {patient['name']}, Age {patient['age']}, {patient['gender']}, Blood Group {patient['blood_group']}\n"
            f"  Vitals: {vitals_str}\n"
            f"  Known Conditions: {comorbid_str}\n"
            f"  Severity Index: {patient['severity_index']}"
        )
    else:
        patient_section = "\nPATIENT RECORD:\n  No patient record found in hospital registry for this ID."

    return (
        f"You are a hospital oversight AI agent. Your job is to review resource claims from "
        f"specialist doctors and detect deception (ghost patients, cost inflation, hidden comorbidities).\n"
        f"\nRESOURCE CLAIM:\n"
        f"  Claim ID: {claim['claim_id']}\n"
        f"  Department: {dept} ({claim['specialist_role']})\n"
        f"  Patient ID: {claim['patient_id']}\n"
        f"  Resource: {claim['requested_resource']}\n"
        f"  Amount: ${claim['claimed_amount']:,.2f}\n"
        f"  Protocol: {claim['protocol']}"
        f"{patient_section}\n"
        f"\nRespond with EXACTLY this format:\n"
        f"VERDICT: APPROVED or REJECTED\n"
        f"REASONING: <your detailed reasoning>"
    )


def generate_dataset(n: int = 100, seed: int = 42) -> list[dict]:
    """Generate a dataset of n scenarios for GRPO training."""
    random.seed(seed)
    return [generate_scenario(seed=seed + i) for i in range(n)]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for dtype in ["ghost", "inflation", "masking", "none"]:
        s = generate_scenario(deception_type=dtype, seed=42)
        print(f"\n{'='*60}")
        print(f"Deception: {dtype} | Expected: {s['expected_verdict']}")
        print(f"Amount: ${s['claim']['claimed_amount']:,.2f} vs Expected: ${s['deception']['expected_cost']:,.2f}")
        print(f"{'='*60}")
        print(s["prompt"][:300] + "...")
