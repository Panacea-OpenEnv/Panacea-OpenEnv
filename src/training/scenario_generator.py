"""
Unified Scenario Generator — The Bridge Between LLM Reasoning and RL Training

Generates adversarial hospital episodes at 3 difficulty levels, using the
existing specialist registry's deception_profile fields.

Difficulty curriculum:
  1 → Static templates only (fast, no API calls — bulk RL training)
  2 → Single-specialist GPT-4o reports (realistic, ~$0.01/episode)
  3 → Multi-specialist + collusion detection (hardest, ~$0.03/episode)

Usage:
  gen = ScenarioGenerator()
  episode = gen.generate(difficulty=1)
  # episode = {patient, vitals, reports[], deception_type, ground_truth_label, prompt}
"""

import json
import os
import random
from typing import Literal

# Reuse existing project components — no duplication
from ..agents.specialists.registry import SPECIALISTS, get_specialist

# ── Patient & clinical data pools ────────────────────────────────────────────

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

# Common drugs for collusion injection
COLLUSION_DRUGS = [
    {"name": "Enoxaparin", "dose": "40mg", "frequency": "once daily", "duration": "7 days"},
    {"name": "Ceftriaxone", "dose": "1g", "frequency": "twice daily", "duration": "5 days"},
    {"name": "Metoprolol", "dose": "50mg", "frequency": "twice daily", "duration": "14 days"},
    {"name": "Pantoprazole", "dose": "40mg", "frequency": "once daily", "duration": "14 days"},
    {"name": "Morphine", "dose": "5mg", "frequency": "every 4 hours PRN", "duration": "3 days"},
]

DeceptionType = Literal["ghost", "inflation", "masking", "collusion", "none"]


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
            COMORBIDITIES_POOL, k=random.randint(0, 4),
        ),
        "severity_index": round(random.uniform(0.2, 1.0), 3),
    }


# ── Static report templates (difficulty=1, no API calls) ─────────────────────

def _generate_static_report(spec_name: str, patient: dict, complaint: str) -> dict:
    """Generate a template-based specialist report. Fast, no API cost."""
    spec = get_specialist(spec_name)
    conditions = spec["conditions"]
    diagnosis = random.choice(conditions)

    return {
        "specialty": spec_name,
        "role": spec["role"],
        "diagnosis": diagnosis.replace("_", " ").title(),
        "severity": random.choice(["low", "medium", "high", "critical"]),
        "medications": [
            {
                "name": random.choice(["Aspirin", "Metformin", "Lisinopril", "Amoxicillin",
                                       "Atorvastatin", "Omeprazole", "Clopidogrel"]),
                "dose": random.choice(["10mg", "25mg", "50mg", "100mg", "500mg"]),
                "frequency": random.choice(["once daily", "twice daily", "three times daily"]),
                "duration": random.choice(["5 days", "7 days", "14 days", "30 days"]),
            }
            for _ in range(random.randint(1, 3))
        ],
        "recommended_tests": random.sample(
            ["CBC", "ECG", "MRI", "CT Scan", "X-Ray", "Blood Culture",
             "Lipid Panel", "Liver Function", "Renal Panel", "Urinalysis"],
            k=random.randint(1, 3),
        ),
        "summary": (
            f"Patient presents with symptoms consistent with {diagnosis.replace('_', ' ')}. "
            f"Based on clinical assessment and vitals review, "
            f"recommend initiating treatment protocol and monitoring."
        ),
        "follow_up": f"Follow up in {random.choice([3, 5, 7, 14])} days.",
    }


# ── Deception injectors ──────────────────────────────────────────────────────

def _inject_ghost() -> dict:
    """Ghost patient: fabricated ID, all reports are fake."""
    ghost_id = f"P{random.randint(9000, 9999)}"
    spec_name = random.choice(list(SPECIALISTS.keys()))
    spec = SPECIALISTS[spec_name]

    # Fabricate a report with no backing patient
    report = {
        "specialty": spec_name,
        "role": spec["role"],
        "diagnosis": "Emergency Resuscitation Required",
        "severity": "critical",
        "medications": [
            {"name": "Epinephrine", "dose": "1mg", "frequency": "stat", "duration": "1 dose"},
        ],
        "recommended_tests": ["Stat CBC", "ABG", "Lactate"],
        "summary": "Patient in acute distress requiring immediate intervention.",
        "follow_up": "ICU admission required.",
    }

    return {
        "patient": None,
        "patient_id": ghost_id,
        "reports": [report],
        "deception_type": "ghost",
        "ground_truth_label": "REJECTED",
        "department": spec_name,
        "claimed_amount": round(random.uniform(30000, 80000), 2),
        "expected_cost": 0.0,  # ghost has no legitimate cost
        "fraud_details": {"ghost_patient_id": ghost_id},
    }


def _inject_inflation(patient: dict, reports: list[dict]) -> dict:
    """Cost inflation: real patient, inflated severity and amount."""
    spec_name = reports[0]["specialty"] if reports else "General Medicine"
    spec = SPECIALISTS.get(spec_name, SPECIALISTS["General Medicine"])

    severity = patient["severity_index"]
    expected_cost = spec.get("base_cost", 15000) * severity
    inflation_factor = round(random.uniform(1.8, 4.0), 2)
    claimed_amount = round(expected_cost * inflation_factor, 2)

    # Inflate severity in reports
    for r in reports:
        r["severity"] = "critical"

    return {
        "patient": patient,
        "patient_id": patient["patient_id"],
        "reports": reports,
        "deception_type": "inflation",
        "ground_truth_label": "REJECTED",
        "department": spec_name,
        "claimed_amount": claimed_amount,
        "expected_cost": round(expected_cost, 2),
        "fraud_details": {
            "inflation_factor": inflation_factor,
            "true_severity": severity,
            "inflated_amount": claimed_amount,
        },
    }


def _inject_masking(patient: dict, reports: list[dict]) -> dict:
    """Comorbidity masking: hide critical conditions from reports."""
    hidden = [c for c in patient["comorbidities"] if c["is_critical"]]
    visible = [c for c in patient["comorbidities"] if not c["is_critical"]]

    if not hidden:
        # Force at least one critical comorbidity to hide
        forced = random.choice([c for c in COMORBIDITIES_POOL if c["is_critical"]])
        patient["comorbidities"].append(forced)
        hidden = [forced]

    # Remove hidden conditions from report summaries
    for r in reports:
        for h in hidden:
            r["summary"] = r["summary"].replace(h["condition"], "")

    spec_name = reports[0]["specialty"] if reports else "General Medicine"
    spec = SPECIALISTS.get(spec_name, SPECIALISTS["General Medicine"])
    severity = patient["severity_index"]
    expected_cost = spec.get("base_cost", 15000) * severity

    return {
        "patient": patient,
        "patient_id": patient["patient_id"],
        "reports": reports,
        "deception_type": "masking",
        "ground_truth_label": "REJECTED",
        "department": spec_name,
        "claimed_amount": round(expected_cost * random.uniform(0.9, 1.1), 2),
        "expected_cost": round(expected_cost, 2),
        "visible_comorbidities": visible,
        "fraud_details": {
            "hidden_conditions": [c["condition"] for c in hidden],
            "hidden_multipliers": [c["multiplier"] for c in hidden],
        },
    }


def _inject_collusion(patient: dict, reports: list[dict]) -> dict:
    """Collusion: 2+ specialists prescribe identical expensive drug."""
    if len(reports) < 2:
        # Need at least 2 reports for collusion — duplicate with different specialty
        available = [s for s in SPECIALISTS.keys() if s != reports[0]["specialty"]]
        second_spec = random.choice(available)
        second_report = _generate_static_report(second_spec, patient, "secondary assessment")
        reports.append(second_report)

    # Pick a shared drug and inject into all reports
    shared_drug = random.choice(COLLUSION_DRUGS)
    for r in reports:
        r["medications"].append(shared_drug)

    spec_name = reports[0]["specialty"]
    spec = SPECIALISTS.get(spec_name, SPECIALISTS["General Medicine"])
    severity = patient["severity_index"]
    expected_cost = spec.get("base_cost", 15000) * severity

    return {
        "patient": patient,
        "patient_id": patient["patient_id"],
        "reports": reports,
        "deception_type": "collusion",
        "ground_truth_label": "REJECTED",
        "department": spec_name,
        "claimed_amount": round(expected_cost * random.uniform(1.0, 1.3), 2),
        "expected_cost": round(expected_cost, 2),
        "fraud_details": {
            "colluding_specialties": [r["specialty"] for r in reports],
            "shared_drug": shared_drug["name"],
        },
    }


# ── Main generator class ─────────────────────────────────────────────────────

class ScenarioGenerator:
    """
    Generates training episodes at 3 difficulty levels.

    Difficulty 1: Static templates (fast, no API calls)
    Difficulty 2: Single-specialist static reports (richer templates)
    Difficulty 3: Multi-specialist + collusion (full complexity)
    """

    def __init__(self, seed: int | None = None, cache_path: str | None = None):
        if seed is not None:
            random.seed(seed)
        self._cache: list[dict] = []
        self._cache_idx: int = 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = [json.loads(line) for line in f if line.strip()]
            print(f"[ScenarioGenerator] Loaded {len(self._cache)} cached GPT-4o episodes from {cache_path}")

    def generate(self, difficulty: int = 1, deception_type: DeceptionType | None = None) -> dict:
        if self._cache:
            episode = self._cache[self._cache_idx % len(self._cache)]
            self._cache_idx += 1
            return episode

        """
        Generate one training episode.

        Returns:
            dict with keys: patient, patient_id, reports[], deception_type,
            ground_truth_label, prompt, department, claimed_amount, expected_cost,
            fraud_details
        """
        # Pick deception type based on difficulty curriculum
        if deception_type is None:
            deception_type = self._pick_deception(difficulty)

        patient_id = f"P{random.randint(1001, 8999)}"
        patient = _generate_patient(patient_id)

        # Generate specialist reports
        num_specialists = 1 if difficulty <= 2 else random.randint(2, 3)
        spec_names = self._pick_specialists(num_specialists, deception_type)
        complaint = random.choice(SPECIALISTS[spec_names[0]]["conditions"]).replace("_", " ")

        reports = [_generate_static_report(s, patient, complaint) for s in spec_names]

        # Inject deception
        if deception_type == "ghost":
            episode = _inject_ghost()
        elif deception_type == "inflation":
            episode = _inject_inflation(patient, reports)
        elif deception_type == "masking":
            episode = _inject_masking(patient, reports)
        elif deception_type == "collusion":
            episode = _inject_collusion(patient, reports)
        else:
            # Clean — no deception
            spec_name = spec_names[0]
            spec = SPECIALISTS[spec_name]
            severity = patient["severity_index"]
            expected_cost = spec.get("base_cost", 15000) * severity
            episode = {
                "patient": patient,
                "patient_id": patient_id,
                "reports": reports,
                "deception_type": "none",
                "ground_truth_label": "APPROVED",
                "department": spec_name,
                "claimed_amount": round(expected_cost * random.uniform(0.9, 1.1), 2),
                "expected_cost": round(expected_cost, 2),
                "fraud_details": {},
            }

        # Format the observation prompt
        episode["prompt"] = self._format_prompt(episode)
        episode["difficulty"] = difficulty
        return episode

    def generate_dataset(self, n: int = 200, difficulty: int = 1, seed: int = 42) -> list[dict]:
        """Generate a dataset of n episodes for training."""
        random.seed(seed)
        return [self.generate(difficulty=difficulty) for _ in range(n)]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pick_deception(self, difficulty: int) -> DeceptionType:
        """Select deception type based on difficulty curriculum."""
        if difficulty == 1:
            # Easy: ghost + inflation + clean
            return random.choices(
                ["ghost", "inflation", "none"],
                weights=[0.25, 0.35, 0.40],
                k=1,
            )[0]
        elif difficulty == 2:
            # Medium: all except collusion
            return random.choices(
                ["ghost", "inflation", "masking", "none"],
                weights=[0.20, 0.25, 0.25, 0.30],
                k=1,
            )[0]
        else:
            # Hard: full adversarial including collusion
            return random.choices(
                ["ghost", "inflation", "masking", "collusion", "none"],
                weights=[0.15, 0.20, 0.20, 0.20, 0.25],
                k=1,
            )[0]

    def _pick_specialists(self, num: int, deception_type: str) -> list[str]:
        """Pick specialist(s), biased toward ones with matching deception_profile."""
        candidates = []
        for name, spec in SPECIALISTS.items():
            profile = spec.get("deception_profile", [])
            if deception_type in profile or deception_type == "none":
                candidates.append(name)

        if not candidates:
            candidates = list(SPECIALISTS.keys())

        return random.sample(candidates, k=min(num, len(candidates)))

    def _format_prompt(self, episode: dict) -> str:
        """Format episode as text prompt for the oversight LLM."""
        patient = episode.get("patient")
        reports = episode.get("reports", [])
        department = episode.get("department", "Unknown")
        claimed = episode.get("claimed_amount", 0)

        # Patient section
        if patient:
            v = patient["vitals"]
            vitals_str = (
                f"HR={v['heart_rate']} bpm, BP={v['bp_systolic']}/{v['bp_diastolic']} mmHg, "
                f"Temp={v['temperature']}C, O2={v['o2_saturation']}%, RR={v['respiratory_rate']}/min"
            )
            comorbid_list = [c["condition"] for c in patient.get("comorbidities", [])]
            # For masking scenarios, only show visible comorbidities
            if episode.get("visible_comorbidities") is not None:
                comorbid_list = [c["condition"] for c in episode["visible_comorbidities"]]
            comorbid_str = ", ".join(comorbid_list) or "None on record"

            patient_section = (
                f"\nPATIENT RECORD:\n"
                f"  ID: {patient['patient_id']}\n"
                f"  Name: {patient['name']}, Age {patient['age']}, {patient['gender']}, "
                f"Blood Group {patient['blood_group']}\n"
                f"  Vitals: {vitals_str}\n"
                f"  Known Conditions: {comorbid_str}\n"
                f"  Severity Index: {patient['severity_index']}"
            )
        else:
            patient_section = (
                f"\nPATIENT RECORD:\n"
                f"  ID: {episode.get('patient_id', 'UNKNOWN')}\n"
                f"  *** NO PATIENT RECORD FOUND IN HOSPITAL REGISTRY ***"
            )

        # Specialist reports section
        reports_section = ""
        for i, r in enumerate(reports, 1):
            meds = ", ".join(m["name"] for m in r.get("medications", []))
            tests = ", ".join(r.get("recommended_tests", []))
            reports_section += (
                f"\n\nSPECIALIST REPORT #{i} — {r['specialty']} ({r['role']}):\n"
                f"  Diagnosis: {r['diagnosis']}\n"
                f"  Severity: {r['severity']}\n"
                f"  Medications: {meds}\n"
                f"  Tests Ordered: {tests}\n"
                f"  Summary: {r['summary']}\n"
                f"  Follow-up: {r['follow_up']}"
            )

        return (
            f"You are a hospital oversight AI agent. Your job is to review resource claims "
            f"from specialist doctors and detect deception.\n"
            f"\nTypes of deception to watch for:\n"
            f"- Ghost patients: fabricated patient IDs with no hospital record\n"
            f"- Cost inflation: claimed amounts far exceeding expected costs for the severity\n"
            f"- Comorbidity masking: critical conditions hidden from the claim\n"
            f"- Collusion: multiple specialists prescribing identical expensive drugs\n"
            f"\nRESOURCE CLAIM:\n"
            f"  Department: {department}\n"
            f"  Claimed Amount: ${claimed:,.2f}"
            f"{patient_section}"
            f"{reports_section}\n"
            f"\nAnalyze the above carefully. Respond with EXACTLY this format:\n"
            f"VERDICT: APPROVED or REJECTED\n"
            f"REASONING: <your detailed reasoning>"
        )


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = ScenarioGenerator(seed=42)

    for diff in [1, 2, 3]:
        ep = gen.generate(difficulty=diff)
        print(f"\n{'='*60}")
        print(f"Difficulty={diff} | Deception={ep['deception_type']} | "
              f"Label={ep['ground_truth_label']}")
        print(f"Reports: {len(ep['reports'])} | Amount: ${ep['claimed_amount']:,.2f}")
        if ep["fraud_details"]:
            print(f"Fraud: {ep['fraud_details']}")
        print(f"{'='*60}")
        print(ep["prompt"][:400] + "...")
