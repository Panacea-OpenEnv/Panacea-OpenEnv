"""
Smart Router Agent — Panacea Hospital

Takes the structured complaint summary from the Intake Nurse and routes to the
right specialist(s) with confidence scores. Much more accurate than routing from
a raw complaint because it uses structured symptom data.
"""

import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .specialists.registry import SPECIALISTS

load_dotenv()

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")

_VALID_SPECIALISTS = list(SPECIALISTS.keys())

_ROUTER_PROMPT = f"""\
You are a Chief Medical Officer routing AI at Panacea Hospital.

Given a structured patient intake summary, output ONLY valid JSON (no explanation, no markdown):

{{
  "specialists": ["PrimarySpecialist", "SecondarySpecialist"],
  "urgency": "critical|high|medium|low",
  "routing_reason": "one sentence explaining why these specialists were chosen"
}}

VALID SPECIALISTS (use ONLY these exact names):
{", ".join(_VALID_SPECIALISTS)}

ROUTING RULES:
- Choose 1-3 specialists maximum
- First specialist = most relevant to chief complaint
- Add a second only if the symptoms clearly cross two systems
- urgency=critical: life-threatening signals (chest pain + sweating, stroke signs, loss of consciousness, severe breathing difficulty)
- urgency=high: significant pain, fever above 39C, vision/hearing sudden loss
- urgency=medium: symptoms lasting days, moderate discomfort
- urgency=low: chronic mild symptoms, general check
- Always prioritize the most specific specialist over General Medicine
"""


async def route_complaint(structured_complaint: dict) -> dict:
    """
    Route a structured intake summary to the right specialist(s).

    Args:
        structured_complaint: Output from intake_nurse.run_intake_interview()

    Returns:
        {"specialists": [...], "urgency": "...", "routing_reason": "..."}
    """
    summary = (
        f"Chief complaint: {structured_complaint.get('chief_complaint', 'unknown')}\n"
        f"Duration: {structured_complaint.get('duration', 'unknown')}\n"
        f"Severity: {structured_complaint.get('severity', 'moderate')}\n"
        f"Alarm signals: {', '.join(structured_complaint.get('severity_signals', [])) or 'none'}\n"
        f"Associated symptoms: {', '.join(structured_complaint.get('associated_symptoms', [])) or 'none'}\n"
        f"Suspected systems: {', '.join(structured_complaint.get('suspected_systems', [])) or 'unspecified'}\n"
        f"Modifiers: {structured_complaint.get('modifiers', {})}"
    )

    try:
        resp = await _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_PROMPT},
                {"role": "user",   "content": summary},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(raw[start:end])
            # Validate specialist names
            result["specialists"] = [
                s for s in result.get("specialists", [])
                if s in _VALID_SPECIALISTS
            ][:3]
            if not result["specialists"]:
                result["specialists"] = ["General Medicine"]
            return result
    except Exception:
        pass

    return {
        "specialists": ["General Medicine"],
        "urgency": structured_complaint.get("severity", "medium"),
        "routing_reason": "Fallback routing — unable to classify complaint precisely.",
    }
