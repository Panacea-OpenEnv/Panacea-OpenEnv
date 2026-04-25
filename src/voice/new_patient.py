"""
New Patient Intake — Panacea Hospital

Registers a new patient in MongoDB with validated inputs,
then launches the full voice diagnosis pipeline.

Usage:
  python -m src.voice.new_patient
"""

import asyncio
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

VALID_BLOOD_GROUPS = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
VALID_GENDERS      = {"M", "F", "OTHER"}


def _ask_required(prompt: str, field: str) -> str:
    """Ask for a required field — re-prompts until non-empty."""
    while True:
        val = input(prompt).strip()
        if val:
            return val
        print(f"  [!] {field} cannot be empty. Please enter a valid value.")


def _ask_name() -> str:
    while True:
        val = input("Full name           : ").strip()
        if not val:
            print("  [!] Name cannot be empty.")
            continue
        if any(ch.isdigit() for ch in val):
            print("  [!] Name should not contain numbers.")
            continue
        if len(val) < 2:
            print("  [!] Name is too short.")
            continue
        return val.title()


def _ask_age() -> int:
    while True:
        val = input("Age                 : ").strip()
        if not val.isdigit():
            print("  [!] Age must be a number (e.g. 25).")
            continue
        age = int(val)
        if age < 1 or age > 120:
            print("  [!] Age must be between 1 and 120.")
            continue
        return age


def _ask_gender() -> str:
    while True:
        val = input("Gender (M / F / Other): ").strip().upper()
        if val in VALID_GENDERS:
            return val
        if not val:
            print("  [!] Please enter M, F, or Other.")
        else:
            print(f"  [!] '{val}' is not valid. Enter M, F, or Other.")


def _ask_blood_group() -> str:
    options = ", ".join(sorted(VALID_BLOOD_GROUPS))
    while True:
        val = input(f"Blood group ({options}): ").strip().upper()
        # Normalise common typos: A positive → A+
        val = val.replace("POSITIVE", "+").replace("NEGATIVE", "-").replace(" ", "")
        if val in VALID_BLOOD_GROUPS:
            return val
        if not val:
            print(f"  [!] Please enter a blood group. Valid options: {options}")
        else:
            print(f"  [!] '{val}' is not a valid blood group. Valid: {options}")


async def register_and_run():
    from src.database.mongo_client import get_collection, PATIENTS
    from src.voice.voice_pipeline import run_voice_session

    print("\n" + "=" * 62)
    print("  PANACEA HOSPITAL — New Patient Registration")
    print("=" * 62)
    print("  Please fill in the details below. All fields are required.\n")

    name        = _ask_name()
    age         = _ask_age()
    gender      = _ask_gender()
    blood_group = _ask_blood_group()

    patient_id = "P" + str(uuid.uuid4().int)[:7].upper()

    patient_doc = {
        "patient_id":    patient_id,
        "name":          name,
        "age":           age,
        "gender":        gender,
        "blood_group":   blood_group,
        "vitals":        {},
        "comorbidities": [],
        "severity_index": 0.0,
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }

    await get_collection(PATIENTS).insert_one(patient_doc)

    print("\n" + "-" * 62)
    print(f"  Registered: {name}  |  Age: {age}  |  {gender}  |  {blood_group}")
    print(f"  Patient ID: {patient_id}")
    print("-" * 62)

    while True:
        mode = input("\nUse microphone? (y/n, default y): ").strip().lower()
        if mode in ("", "y", "n"):
            break
        print("  [!] Enter y for microphone or n for keyboard.")
    text_mode = mode == "n"

    print("\n" + "=" * 62)
    print(f"  Starting diagnosis for {name}...")
    print("=" * 62 + "\n")

    await run_voice_session(
        patient_id=patient_id,
        text_mode=text_mode,
    )


if __name__ == "__main__":
    asyncio.run(register_and_run())
