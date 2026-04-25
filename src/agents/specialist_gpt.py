"""
GPT-4o Specialist Agents
Each of the 26 specialists uses this module to:
  1. Load their domain system prompt
  2. Inject returning patient history from MongoDB
  3. Stream responses token-by-token (near-zero perceived latency)
  4. Ask specialty-specific questions across 3-5 turns
  5. Generate structured final diagnosis + medications
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .specialists.registry import get_specialist
from ..database.mongo_client import (
    get_patient, get_patient_vitals, get_patient_comorbidities,
    get_patient_history, save_specialist_report,
)

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TURNS = int(os.getenv("MAX_SPECIALIST_TURNS", "5"))

# ── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(
    spec_name: str,
    patient: dict,
    vitals: dict,
    comorbidities: list[dict],
    history: list[dict],
) -> str:
    spec = get_specialist(spec_name)
    role = spec["role"]
    conditions = ", ".join(spec["conditions"][:6])
    protocols  = ", ".join(spec["protocols"][:3])

    # Format vitals
    v = vitals or {}
    vitals_str = (
        f"HR={v.get('heart_rate','?')} bpm, "
        f"BP={v.get('bp_systolic','?')}/{v.get('bp_diastolic','?')} mmHg, "
        f"Temp={v.get('temperature','?')}°C, "
        f"O2={v.get('o2_saturation','?')}%, "
        f"RR={v.get('respiratory_rate','?')}/min"
    ) if v else "Not available"

    # Format comorbidities
    comorbid_str = ", ".join(
        f"{c['condition']} {'(CRITICAL)' if c.get('is_critical') else ''}"
        for c in comorbidities
    ) if comorbidities else "None on record"

    # Format past consultation history
    history_str = ""
    if history:
        entries = []
        for h in history:
            s = h.get("final_summary", {})
            entries.append(
                f"- {h.get('created_at','')}: {s.get('primary_diagnosis','unknown')} "
                f"| Meds: {', '.join(m.get('name','') for m in s.get('all_medications',[])[:3])}"
            )
        history_str = "PREVIOUS VISITS:\n" + "\n".join(entries)
    else:
        history_str = "PREVIOUS VISITS: First visit — no history on record."

    return f"""You are Dr. {patient.get('name', 'Patient')}'s {role} at Panacea Hospital.

PATIENT: {patient.get('name','Unknown')}, Age {patient.get('age','?')}, {patient.get('gender','?')}, Blood Group {patient.get('blood_group','?')}
CURRENT VITALS: {vitals_str}
KNOWN CONDITIONS: {comorbid_str}
{history_str}

YOUR SPECIALTY: {spec_name}
YOU TREAT: {conditions}
YOUR PROTOCOLS: {protocols}

INSTRUCTIONS:
- You are a human doctor speaking to a patient. Be warm, conversational, empathetic, and professional.
- NO MARKDOWN. No **bold**, no *italics*, no bullet points, no numbered lists.
- NEVER use curly braces {{ or }} in your conversational replies. They are reserved ONLY for your final JSON assessment.
- Keep your questions short and natural. Ask only ONE thing at a time.
- While you are still diagnosing, output ONLY plain spoken text. Absolutely no JSON.
- After asking enough questions (usually 2-4 turns), deliver your final assessment as a SINGLE message containing ONLY this JSON (no text before or after it):

{{
  "diagnosis": "primary diagnosis here",
  "severity": "critical|high|medium|low",
  "medications": [
    {{"name": "drug name", "dose": "dose", "frequency": "frequency", "duration": "duration"}}
  ],
  "recommended_tests": ["test1", "test2"],
  "follow_up": "follow-up instruction",
  "summary": "2-3 sentence plain language summary for the patient",
  "ready": true
}}

Start by warmly greeting the patient and asking your first most important question."""


# ── Streaming specialist conversation ─────────────────────────────────────────

async def stream_specialist_response(
    messages: list[dict],
    on_token=None,
) -> str:
    """
    Stream GPT-4o response token by token.
    on_token(str) callback fires for each token — used by TTS to speak immediately.
    Returns full response string.
    """
    full_response = ""
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=0.4,
        max_tokens=600,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        if on_token and delta:
            on_token(delta)

    return full_response


async def run_specialist_consultation(
    spec_name: str,
    patient_id: str,
    session_id: str,
    initial_complaint: str,
    on_token=None,
    on_turn_complete=None,
    voice_mode: bool = False,
    patient_override: dict | None = None,
) -> dict:
    """
    Full multi-turn specialist consultation.

    Args:
        spec_name:         e.g. "Cardiology"
        patient_id:        e.g. "P1001"
        session_id:        unique session UUID
        initial_complaint: patient's first spoken complaint (from voice STT)
        on_token:          callback(token_str) for streaming TTS
        on_turn_complete:  callback(turn_dict) for terminal display
        patient_override:  dict containing 'patient', 'vitals', 'comorbids', 'history' to skip DB

    Returns:
        dict with full conversation, diagnosis, medications
    """
    # ── Load patient context from MongoDB or override ────────────────────────
    if patient_override:
        patient   = patient_override.get("patient", {})
        vitals    = patient_override.get("vitals", {})
        comorbids = patient_override.get("comorbids", [])
        history   = patient_override.get("history", [])
    else:
        patient      = await get_patient(patient_id) or {}
        vitals       = await get_patient_vitals(patient_id)
        comorbids    = await get_patient_comorbidities(patient_id)
        history      = await get_patient_history(patient_id, limit=3)

    system_prompt = build_system_prompt(spec_name, patient, vitals, comorbids, history)

    messages: list[dict] = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": initial_complaint},
    ]

    conversation: list[dict] = [
        {"speaker": "patient", "text": initial_complaint, "timestamp": _now()}
    ]

    final_assessment: dict = {}
    spec_data = get_specialist(spec_name)

    for turn in range(MAX_TURNS):
        # ── GPT-4o streams response ───────────────────────────────────────────
        response_text = await stream_specialist_response(messages, on_token=on_token)

        conversation.append({
            "speaker":   spec_data["role"],
            "text":      response_text,
            "timestamp": _now(),
        })

        if on_turn_complete:
            on_turn_complete({
                "turn":      turn + 1,
                "specialist": spec_name,
                "role":      spec_data["role"],
                "response":  response_text,
            })

        messages.append({"role": "assistant", "content": response_text})

        # ── Check if specialist is done (returned JSON with ready:true) ───────
        assessment = _extract_assessment(response_text)
        if assessment and assessment.get("ready"):
            final_assessment = assessment
            break

        # ── Get patient reply (from voice pipeline or simulated) ─────────────
        if voice_mode:
            # Voice pipeline injects real patient speech via inject_patient_reply()
            patient_reply = await _wait_for_patient_reply()
        else:
            patient_reply = f"[simulated patient answer to turn {turn + 1}]"

        conversation.append({
            "speaker":   "patient",
            "text":      patient_reply,
            "timestamp": _now(),
        })
        messages.append({"role": "user", "content": patient_reply})

    # ── Build report document ─────────────────────────────────────────────────
    report = {
        "session_id":   session_id,
        "patient_id":   patient_id,
        "specialty":    spec_name,
        "role":         spec_data["role"],
        "conversation": conversation,
        "diagnosis":    final_assessment.get("diagnosis", "Assessment incomplete"),
        "severity":     final_assessment.get("severity", "unknown"),
        "medications":  final_assessment.get("medications", []),
        "recommended_tests": final_assessment.get("recommended_tests", []),
        "follow_up":    final_assessment.get("follow_up", ""),
        "summary":      final_assessment.get("summary", ""),
        "created_at":   _now(),
    }

    await save_specialist_report(report)
    return report


# ── Patient reply queue (used in voice pipeline mode) ────────────────────────
_patient_reply_queue: asyncio.Queue = asyncio.Queue()

async def _wait_for_patient_reply() -> str:
    """Called by specialist — blocks until voice pipeline puts patient reply."""
    return await _patient_reply_queue.get()

async def inject_patient_reply(text: str):
    """Called by voice pipeline after Whisper transcribes patient's speech."""
    await _patient_reply_queue.put(text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_assessment(text: str) -> dict | None:
    """Parse JSON assessment block from GPT-4o response if present.
    
    Guards against false positives:
      - JSON substring must be at least 50 chars (a real assessment is long)
      - Parsed result must be a dict containing the 'ready' key
    """
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        json_str = text[start:end]
        if len(json_str) < 50:
            return None
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and "ready" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Quick test (no voice, no MongoDB needed) ──────────────────────────────────

async def _test():
    print(f"\nTesting GPT-4o specialist: Cardiology")
    print("=" * 50)

    tokens_seen = []

    def on_token(t):
        print(t, end="", flush=True)
        tokens_seen.append(t)

    def on_turn(turn_info):
        print(f"\n[Turn {turn_info['turn']} complete — {turn_info['role']}]")

    report = await run_specialist_consultation(
        spec_name="Cardiology",
        patient_id="P1001",
        session_id="test-session-001",
        initial_complaint="I have severe chest pain radiating to my left arm and I am sweating a lot",
        on_token=on_token,
        on_turn_complete=on_turn,
        voice_mode=False,   # test mode — simulates patient replies, no mic needed
    )

    print(f"\n\nDiagnosis: {report['diagnosis']}")
    print(f"Severity:  {report['severity']}")
    print(f"Meds:      {report['medications']}")
    print(f"Tests:     {report['recommended_tests']}")
    print(f"Follow-up: {report['follow_up']}")
    print(f"\nSummary: {report['summary']}")

if __name__ == "__main__":
    asyncio.run(_test())
