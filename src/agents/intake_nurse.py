"""
Intake Nurse Agent — Panacea Hospital

Interviews the patient with 3-5 clarifying questions before routing to specialists.
Transforms a vague complaint into a structured symptom summary that the router uses.

Example:
  Patient: "my eyes are burning"
  Nurse:   "How many days has this been going on?"
  Patient: "about 3 days"
  Nurse:   "Are your eyes red or watering?"
  ...
  Output:  {chief_complaint, duration, severity_signals, modifiers, suspected_systems}
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")

_NURSE_SYSTEM_PROMPT = """\
You are Priya, a warm and professional intake nurse at Panacea Hospital.
Your job is to interview the patient with 3 to 5 short clarifying questions, then produce a structured summary.

STRICT RULES:
- Ask EXACTLY ONE short question per turn. Never combine two questions.
- Be empathetic and use simple non-medical language.
- NO MARKDOWN. No bullet points, no bold, no lists.
- Do NOT diagnose or suggest treatments.
- The patient has already been greeted — do NOT say hello or introduce yourself again.
  Start by briefly acknowledging their complaint and immediately asking your first question.
- After 3 to 5 questions (no more), you MUST stop asking and output ONLY the JSON below.
  Do NOT ask more than 5 questions under any circumstance.
- If you already have enough information after 3 questions, finalize immediately.

WHEN DONE — output ONLY this JSON with no text before or after it:

{
  "chief_complaint": "concise description of the main problem",
  "duration": "how long the symptom has been present",
  "severity": "mild|moderate|severe",
  "severity_signals": ["any alarm symptoms like fever, bleeding, chest pain, etc."],
  "associated_symptoms": ["other symptoms the patient mentioned"],
  "modifiers": {"worse_with": "...", "better_with": "...", "location": "..."},
  "suspected_systems": ["body system(s) likely involved, e.g. musculoskeletal, ophthalmology"],
  "ready": true
}"""


async def run_intake_interview(
    initial_complaint: str,
    on_token=None,
    on_turn_complete=None,
    voice_mode: bool = False,
) -> dict:
    """
    Run a multi-turn intake nurse interview.

    Args:
        initial_complaint: Patient's opening statement
        on_token:          Callback(str) for streaming TTS
        on_turn_complete:  Callback(dict) for display
        voice_mode:        If True, waits for real patient voice reply

    Returns:
        Structured complaint dict with chief_complaint, duration, severity, etc.
    """
    messages = [
        {"role": "system", "content": _NURSE_SYSTEM_PROMPT},
        {"role": "user",   "content": initial_complaint},
    ]

    max_turns = 6

    for turn in range(max_turns):
        # Stream nurse response
        full_response = ""
        stream = await _client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            stream=True,
            temperature=0.3,
            max_tokens=300,
        )

        json_detected = False
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta

            if on_token and delta:
                # Suppress TTS if this looks like a JSON response
                so_far = full_response.lstrip()
                if so_far.startswith("{"):
                    json_detected = True
                if not json_detected:
                    on_token(delta)

        if on_turn_complete:
            on_turn_complete({"turn": turn + 1, "response": full_response})

        messages.append({"role": "assistant", "content": full_response})

        # Check if nurse is done (returned structured JSON)
        result = _extract_structured(full_response)
        if result and result.get("ready"):
            return result

        # Get patient reply
        if voice_mode:
            patient_reply = await _wait_for_patient_reply()
        else:
            patient_reply = f"[simulated patient answer {turn + 1}]"

        messages.append({"role": "user", "content": patient_reply})

    # Fallback if nurse never returned structured JSON
    return {
        "chief_complaint": initial_complaint,
        "duration": "unknown",
        "severity": "moderate",
        "severity_signals": [],
        "associated_symptoms": [],
        "modifiers": {},
        "suspected_systems": [],
        "ready": True,
    }


def _extract_structured(text: str) -> dict | None:
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        parsed = json.loads(text[start:end])
        if isinstance(parsed, dict) and "ready" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


#  Patient reply queue (shared with voice pipeline) 
_intake_reply_queue: asyncio.Queue = asyncio.Queue()

async def _wait_for_patient_reply() -> str:
    return await _intake_reply_queue.get()

async def inject_intake_reply(text: str):
    """Called by voice pipeline after STT transcribes patient's reply to nurse."""
    await _intake_reply_queue.put(text)
