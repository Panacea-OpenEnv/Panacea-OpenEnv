"""
GPT-4o Report Harvester
Generates a JSONL cache of real specialist reports (with deception injected).
Run ONCE before training. Training then samples from the cache — no API calls during RL.
Usage:
    python -m src.training.gpt4o_harvester
"""

import asyncio
import json
import os
import re
import random
from tqdm import tqdm
from dotenv import load_dotenv

from .scenario_generator import (
    _generate_patient, 
    _inject_inflation, 
    _inject_collusion,
    _inject_ghost,
    _inject_masking
)
from ..agents.specialist_gpt import build_system_prompt, stream_specialist_response
from ..agents.specialists.registry import SPECIALISTS

load_dotenv()


def _strip_json_fence(raw: str) -> str:
    """Strip markdown code fences from GPT-4o output.
    GPT-4o returns ```json\n{...}\n``` ~30% of the time even when told not to.
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    stripped = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    stripped = re.sub(r'```\s*$', '', stripped.strip(), flags=re.MULTILINE)
    return stripped.strip()

async def harvest_reports(num_episodes: int = 100, out_path: str = "data/gpt4o_reports.jsonl"):
    os.makedirs("data", exist_ok=True)
    
    with open(out_path, "w") as f:
        for i in tqdm(range(num_episodes), desc="Harvesting Episodes"):
            pid = f"P{random.randint(1000, 9999)}"
            patient = _generate_patient(pid)
            
            # Determine deception type
            deception = random.choice(["none", "none", "inflation", "collusion", "ghost", "masking"])
            
            # 1. Generate core report using stream_specialist_response (single-shot)
            spec_name = random.choice(list(SPECIALISTS.keys()))
            vitals = patient.get("vitals", {})
            comorbids = patient.get("comorbidities", [])
            
            # Instruct model to return raw JSON
            system_prompt = build_system_prompt(spec_name, patient, vitals, comorbids, [])
            system_prompt += "\n\nCRITICAL INSTRUCTION: You must respond ONLY with a raw JSON object matching the assessment format. Do not use markdown code blocks (```json). Ensure 'medications' is a list of dicts with 'name'."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "I am the patient. I have symptoms. Please provide my assessment as a JSON object."}
            ]
            
            try:
                raw_json = await stream_specialist_response(messages)
                cleaned = _strip_json_fence(raw_json)
                report = json.loads(cleaned)
                report["specialty"] = spec_name
                report["role"] = SPECIALISTS[spec_name]["role"]
            except json.JSONDecodeError as e:
                print(f"[Episode {i}] JSON parse failed — raw output was: {raw_json[:200]!r}")
                print(f"  Error: {e}")
                continue
            except Exception as e:
                print(f"[Episode {i}] GPT-4o call failed: {e}")
                continue
                
            reports = [report]
            
            # 2. Inject deception
            if deception == "ghost":
                scenario = _inject_ghost()
                scenario["patient"] = None
                # Override the report with the one we just got (or use the ghost's fake one)
                scenario["reports"] = reports
            elif deception == "inflation":
                scenario = _inject_inflation(patient, reports)
            elif deception == "masking":
                scenario = _inject_masking(patient, reports)
            elif deception == "collusion":
                scenario = _inject_collusion(patient, reports)
            else:
                # Legitimate scenario
                spec = SPECIALISTS.get(spec_name)
                expected_cost = spec.get("base_cost", 15000) * patient["severity_index"]
                scenario = {
                    "patient": patient,
                    "patient_id": patient["patient_id"],
                    "reports": reports,
                    "deception_type": "none",
                    "ground_truth_label": "APPROVED",
                    "department": spec_name,
                    "claimed_amount": round(expected_cost * random.uniform(0.9, 1.1), 2),
                    "expected_cost": round(expected_cost, 2),
                    "fraud_details": {},
                }
                
            reports = scenario["reports"]
            patient_from_db = scenario["patient"]
            
            # 3. Format prompt exactly as oversight expects
            resource_requests = []
            for r in reports:
                resource_requests.append({
                    "specialist": r.get("specialty"),
                    "resources": [m.get("name", "") for m in r.get("medications", [])] + r.get("recommended_tests", []),
                    "severity": 0.8 if r.get("severity", "") == "critical" else 0.5
                })
                
            total_claimed = scenario.get("claimed_amount", 0)
            
            reports_text = ""
            for idx, r in enumerate(reports, 1):
                reports_text += (
                    f"\nSPECIALIST REPORT #{idx} — {r.get('specialty', r.get('specialist', ''))} ({r.get('role', '')}):\n"
                    f"  Severity: {r.get('severity_label', r.get('severity', ''))}\n"
                    f"  Resources: {r.get('requested_resources', [m.get('name', '') for m in r.get('medications', [])])}\n"
                    f"  Assessment: {r.get('summary', r.get('assessment', ''))[:200]}"
                )

            prompt = (
                f"You are a hospital oversight AI agent. Analyze this claim for deception.\n"
                f"\nRESOURCE CLAIM:\n"
                f"  Patient ID: {pid}\n"
                f"  Specialists: {[r.get('specialty', r.get('specialist', '')) for r in reports]}\n"
                f"  Estimated Claimed Amount: ${total_claimed:,.2f}"
                f"{reports_text}\n"
                f"\nRespond with:\n"
                f"VERDICT: APPROVED or REJECTED\n"
                f"REASONING: <your detailed reasoning>"
            )
            
            episode_data = {
                "patient": patient_from_db,
                "reports": reports,
                "resource_requests": resource_requests,
                "deception_type": scenario["deception_type"],
                "ground_truth_label": scenario["ground_truth_label"],
                "prompt": prompt
            }
            
            f.write(json.dumps(episode_data) + "\n")

if __name__ == "__main__":
    asyncio.run(harvest_reports(50)) # Generating 50 episodes initially for quick test
