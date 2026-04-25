"""
Consult Bridge — Cross-Specialist Consultation

When a specialist determines they need a second opinion (e.g., Ophthalmologist
notices diabetic signs and needs Endocrinology input), this module:
  1. Runs a brief 1-2 question consultation with the second specialist
  2. Returns a structured opinion that gets injected back into the
     primary specialist's context so they can finalize with full information.

The patient may be asked 1-2 targeted questions by the consulting specialist.
The primary specialist never loses their thread — they resume after the consult.
"""

import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .specialists.registry import get_specialist, SPECIALISTS

load_dotenv()

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")

_CONSULT_SYSTEM = """\
You are a consulting specialist called in for a brief second opinion.
The primary doctor has already spoken to the patient. You are given a summary of findings.

Your job:
1. Ask the patient ONE short targeted question relevant to your specialty.
2. Based on their answer, provide your consultation opinion.

Output your final opinion as ONLY this JSON (no text before or after):
{
  "consulting_specialty": "your specialty name",
  "opinion": "your clinical finding in 1-2 sentences",
  "additional_medications": [{"name": "...", "dose": "...", "frequency": "...", "duration": "..."}],
  "additional_tests": ["test1", "test2"],
  "concern": "brief concern or 'none'"
}
"""

_DETECT_CONSULT_PROMPT = """\
You are analyzing a specialist's message to a patient.
Does this message indicate the specialist wants to call in another specialist for a second opinion?

If yes, extract: which specialty to consult and why.
Output ONLY JSON:
{"needs_consult": true, "consult_specialty": "SpecialtyName", "reason": "why"}
or
{"needs_consult": false}

The message to analyze:
"""


async def detect_consult_request(specialist_message: str) -> dict | None:
    """
    Check if a specialist's message contains a consult request signal.
    Returns consult info dict or None.
    """
    # Fast keyword check first (avoid GPT call for every token)
    lower = specialist_message.lower()
    consult_keywords = [
        "consult", "second opinion", "refer", "bring in", "colleague",
        "endocrinologist", "cardiologist", "neurologist", "specialist",
        "another doctor", "check with"
    ]
    if not any(kw in lower for kw in consult_keywords):
        return None

    # Ask GPT to confirm and extract
    try:
        resp = await _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "user", "content": _DETECT_CONSULT_PROMPT + specialist_message},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(raw[start:end])
            if result.get("needs_consult") and result.get("consult_specialty") in SPECIALISTS:
                return result
    except Exception:
        pass
    return None


async def run_consult(
    consult_specialty: str,
    primary_specialty: str,
    patient: dict,
    primary_findings: str,
    on_token=None,
    voice_mode: bool = False,
    get_patient_reply=None,
) -> dict:
    """
    Run a brief cross-specialist consultation (1-2 patient questions max).

    Args:
        consult_specialty:  e.g. "Endocrinology"
        primary_specialty:  e.g. "Ophthalmology" (who is calling)
        patient:            Patient record dict
        primary_findings:   Summary of what primary specialist has found so far
        on_token:           TTS streaming callback
        voice_mode:         If True, uses real STT for patient reply
        get_patient_reply:  Async callable that returns patient's spoken reply

    Returns:
        Consult opinion dict with additional_medications, additional_tests, opinion
    """
    spec = get_specialist(consult_specialty)
    name = patient.get("name", "Patient")
    age  = patient.get("age", "?")

    system_prompt = (
        f"{_CONSULT_SYSTEM}\n\n"
        f"YOU ARE: {spec['role']} ({consult_specialty})\n"
        f"CALLED BY: {primary_specialty}\n"
        f"PATIENT: {name}, Age {age}\n"
        f"PRIMARY FINDINGS: {primary_findings}\n"
        f"YOU TREAT: {', '.join(spec['conditions'][:5])}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Please ask your one targeted question to the patient."},
    ]

    # Turn 1: consulting specialist asks patient one question
    full_response = ""
    stream = await _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        stream=True,
        temperature=0.3,
        max_tokens=200,
    )

    json_detected = False
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        if on_token and delta:
            so_far = full_response.lstrip()
            if so_far.startswith("{"):
                json_detected = True
            if not json_detected:
                on_token(delta)

    messages.append({"role": "assistant", "content": full_response})

    # Check if they already returned JSON (decided without asking patient)
    result = _extract_opinion(full_response)
    if result:
        return result

    # Turn 2: get patient reply and produce final opinion
    if voice_mode and get_patient_reply:
        patient_reply = await get_patient_reply()
    else:
        patient_reply = "I am not sure, please advise."

    messages.append({"role": "user", "content": patient_reply})
    messages.append({"role": "user", "content": "Now provide your consultation opinion as JSON."})

    opinion_response = ""
    stream2 = await _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
        max_tokens=400,
    )
    async for chunk in stream2:
        delta = chunk.choices[0].delta.content or ""
        opinion_response += delta

    result = _extract_opinion(opinion_response)
    if result:
        result["consulting_specialty"] = consult_specialty
        return result

    return {
        "consulting_specialty": consult_specialty,
        "opinion": "No additional concerns from consulting specialist.",
        "additional_medications": [],
        "additional_tests": [],
        "concern": "none",
    }


def _extract_opinion(text: str) -> dict | None:
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= start or (end - start) < 30:
            return None
        parsed = json.loads(text[start:end])
        if isinstance(parsed, dict) and "opinion" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def merge_consult_into_report(report: dict, consult_opinion: dict) -> dict:
    """
    Merge a consult opinion into an existing specialist report.
    Adds extra medications, tests, and appends consult note to summary.
    """
    if not consult_opinion:
        return report

    extra_meds  = consult_opinion.get("additional_medications", [])
    extra_tests = consult_opinion.get("additional_tests", [])
    opinion     = consult_opinion.get("opinion", "")
    consult_by  = consult_opinion.get("consulting_specialty", "Consulting Specialist")

    existing_med_names = {m.get("name", "") for m in report.get("medications", [])}
    for m in extra_meds:
        if m.get("name") not in existing_med_names:
            report.setdefault("medications", []).append(m)

    existing_tests = set(report.get("recommended_tests", []))
    for t in extra_tests:
        if t not in existing_tests:
            report.setdefault("recommended_tests", []).append(t)

    if opinion:
        report["summary"] = (
            report.get("summary", "") +
            f" [{consult_by} consult: {opinion}]"
        )

    report.setdefault("consults", []).append(consult_opinion)
    return report
