"""
Voice Pipeline — Panacea Patient Intake

Full end-to-end voice conversation loop:
  1. Greet patient (TTS)
  2. Record initial complaint (STT via faster-whisper)
  3. GPT-4o triage → decide which specialists to activate
  4. Per specialist: multi-turn voice conversation
       - GPT-4o streams tokens → TTS speaks sentence-by-sentence
       - Patient speaks → STT transcribes → injected into specialist queue
  5. Synthesize all specialist reports
  6. Save session to MongoDB (patient_consultations + medical_summaries)
  7. TTS reads final summary aloud

Usage:
  # Voice mode (microphone required)
  python -m src.voice.voice_pipeline P1001

  # Text mode (keyboard input — no microphone required)
  python -m src.voice.voice_pipeline P1001 --text
"""

import asyncio
import json
import uuid
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .stt import stt
from .tts import tts
from ..agents.specialist_gpt import run_specialist_consultation, inject_patient_reply
from ..database.mongo_client import get_patient, save_consultation, save_medical_summary
from ..utils.terminal_display import display

load_dotenv()

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")

_TRIAGE_PROMPT = """\
You are a Chief Medical Officer triage AI at Panacea Hospital.
Given a patient's spoken complaint, output ONLY valid JSON (no explanation):

{
  "specialists": ["Specialty1", "Specialty2"],
  "urgency": "critical|high|medium|low",
  "symptoms": ["symptom1", "symptom2"]
}

Choose 1-3 specialties from EXACTLY this list (no other values allowed):
Cardiology, Neurology, Neurosurgery, Orthopedics, Pediatrics, Gynecology,
Obstetrics, Dermatology, Ophthalmology, Otolaryngology, Gastroenterology,
Pulmonology, Nephrology, Urology, Endocrinology, Oncology, Hematology,
Rheumatology, Psychiatry, Radiology, Anesthesiology, General Medicine,
Pathology, Plastic Surgery, Vascular Surgery, Infectious Disease

Rules:
- chest pain / left arm pain / sweating → Cardiology (urgency=critical)
- stroke / facial droop / speech slur → Neurology (urgency=critical)
- difficulty breathing → Pulmonology or Cardiology (urgency=critical)
- loss of consciousness → Neurology + Anesthesiology (urgency=critical)
- high fever / infection → Infectious Disease or General Medicine
- Always include at least one specialist
- For unclear complaints use General Medicine"""


# ─────────────────────────────────────────────────────────────────────────────
# Triage
# ─────────────────────────────────────────────────────────────────────────────

async def _triage_complaint(complaint: str) -> dict:
    """Call GPT-4o to classify complaint → specialists + urgency."""
    try:
        resp = await _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _TRIAGE_PROMPT},
                {"role": "user",   "content": f"Patient says: {complaint}"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except Exception as exc:
        display.error("TRIAGE", str(exc))
    return {"specialists": ["General Medicine"], "urgency": "medium", "symptoms": []}


# ─────────────────────────────────────────────────────────────────────────────
# Single-specialist voice loop
# ─────────────────────────────────────────────────────────────────────────────

async def _run_voice_specialist(
    spec_name: str,
    patient_id: str,
    session_id: str,
    complaint: str,
    text_mode: bool = False,
) -> dict:
    """
    Run one specialist consultation in voice mode.

    Architecture:
      - specialist_task: runs GPT-4o consultation, streams tokens to TTS,
        then awaits patient reply via _patient_reply_queue
      - inject_task: after each specialist turn, waits for TTS to finish,
        records patient speech (or stdin), injects text into the reply queue

    The two tasks communicate via:
      - turn_signal (asyncio.Queue): specialist signals "I finished speaking"
      - inject_patient_reply(): voice pipeline pushes patient text into
        specialist_gpt._patient_reply_queue
    """
    loop        = asyncio.get_event_loop()
    turn_signal: asyncio.Queue = asyncio.Queue()

    # ── Callbacks passed into specialist_gpt ──────────────────────────────────

    def on_token(token: str):
        tts.stream_token(token)

    def on_turn_done(turn_info: dict):
        # Called synchronously from within the specialist coroutine after
        # each GPT-4o response, right before _wait_for_patient_reply().
        # put_nowait is safe here because we're still in the event loop.
        turn_signal.put_nowait(turn_info)
        display.specialist_turn_done(spec_name, turn_info["turn"])

    # ── Specialist consultation task ───────────────────────────────────────────

    spec_task = asyncio.create_task(
        run_specialist_consultation(
            spec_name       = spec_name,
            patient_id      = patient_id,
            session_id      = f"{session_id}_{spec_name.replace(' ', '_')}",
            initial_complaint = complaint,
            on_token        = on_token,
            on_turn_complete= on_turn_done,
            voice_mode      = True,
        )
    )

    # ── Patient reply injector task ────────────────────────────────────────────

    async def inject_loop():
        while not spec_task.done():
            # Wait for specialist to finish a turn (signals via on_turn_done)
            try:
                await asyncio.wait_for(turn_signal.get(), timeout=120.0)
            except asyncio.TimeoutError:
                display.error(spec_name, "Turn timeout — ending consultation")
                break

            # Flush partial sentence buffer, wait for TTS to finish speaking
            tts.flush_buffer()
            await loop.run_in_executor(None, tts.wait_until_done)

            # If specialist finished (final assessment) while TTS was playing, stop
            if spec_task.done():
                break

            # Record patient's spoken reply
            if text_mode:
                print(f"\n[{spec_name}] Your reply: ", end="", flush=True)
                reply = input().strip()
            else:
                display.patient_listening()
                reply = await loop.run_in_executor(None, stt.record_until_silence)

            reply = reply or "I'm not sure, could you explain more?"
            display.patient_speech(reply)
            await inject_patient_reply(reply)

    inject_task = asyncio.create_task(inject_loop())

    try:
        report = await spec_task
    finally:
        inject_task.cancel()
        try:
            await inject_task
        except asyncio.CancelledError:
            pass

    # Ensure any leftover TTS is spoken before returning
    tts.flush_buffer()
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Report synthesis
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}


def _synthesize_reports(reports: list[dict], patient: dict) -> dict:
    """Merge all specialist reports into one final medical summary."""
    if not reports:
        return {}

    lead = max(reports, key=lambda r: _SEVERITY_RANK.get(r.get("severity", "unknown"), 0))

    all_meds:  list[dict] = []
    all_tests: list[str]  = []
    seen_meds:  set[str]  = set()
    seen_tests: set[str]  = set()

    for r in reports:
        for m in r.get("medications", []):
            name = m.get("name", "")
            if name and name not in seen_meds:
                seen_meds.add(name)
                all_meds.append(m)
        for t in r.get("recommended_tests", []):
            if t and t not in seen_tests:
                seen_tests.add(t)
                all_tests.append(t)

    summaries = [r.get("summary", "") for r in reports if r.get("summary")]
    combined  = " ".join(summaries[:2])

    plain = (
        f"Based on our specialist assessments, your primary condition is "
        f"{lead.get('diagnosis', 'under evaluation')}. "
        + (
            f"We are prescribing {', '.join(m['name'] for m in all_meds[:3])}. "
            if all_meds else ""
        )
        + "Please follow your follow-up instructions carefully and rest well."
    )

    return {
        "patient_id":          patient.get("patient_id", ""),
        "patient_name":        patient.get("name", ""),
        "primary_diagnosis":   lead.get("diagnosis", ""),
        "primary_specialty":   lead.get("specialty", ""),
        "severity":            lead.get("severity", "medium"),
        "all_medications":     all_meds,
        "all_tests":           all_tests,
        "specialist_count":    len(reports),
        "specialists_involved":[r.get("specialty", "") for r in reports],
        "detailed_summary":    combined,
        "plain_language":      plain,
        "follow_up":           lead.get("follow_up", ""),
        "created_at":          datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main session runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_voice_session(
    patient_id: str = "P1001",
    text_mode:  bool = False,
    preset_complaint: str = "",
) -> dict:
    """
    Full patient intake voice session.

    Args:
        patient_id: MongoDB patient ID (must exist in 'patients' collection)
        text_mode:  If True, use stdin instead of microphone (for testing)

    Returns:
        Final medical summary dict saved to MongoDB
    """
    session_id = str(uuid.uuid4())[:12]
    loop       = asyncio.get_event_loop()

    # ── Startup ────────────────────────────────────────────────────────────────
    if not text_mode:
        stt.load()
    tts.start()

    patient = await get_patient(patient_id) or {
        "patient_id": patient_id,
        "name":       "Patient",
    }
    display.info(f"Starting voice session for {patient.get('name')} ({patient_id})")

    # ── Greeting ───────────────────────────────────────────────────────────────
    tts.speak(
        f"Welcome to Panacea Hospital. "
        f"I am your virtual intake coordinator. "
        f"Please describe your main symptoms and how you are feeling today."
    )
    await loop.run_in_executor(None, tts.wait_until_done)

    # ── Initial complaint ──────────────────────────────────────────────────────
    if preset_complaint:
        complaint = preset_complaint
        text_mode = True   # preset complaint → text mode for all specialist Q&A too
    elif text_mode:
        print("\nEnter patient complaint: ", end="", flush=True)
        complaint = input().strip()
    else:
        display.patient_listening()
        complaint = await loop.run_in_executor(None, stt.record_until_silence)

    complaint = complaint or "I have a general health concern I would like to discuss."
    display.patient_speech(complaint)

    # ── GPT-4o Triage ──────────────────────────────────────────────────────────
    tts.speak("Thank you. I am connecting you to the right specialists now. One moment please.")
    routing = await _triage_complaint(complaint)

    specialists: list[str] = routing.get("specialists", ["General Medicine"])[:3]
    urgency:     str       = routing.get("urgency", "medium")
    symptoms:    list[str] = routing.get("symptoms", [])

    display.session_header(
        patient_id = patient_id,
        session_id = session_id,
        urgency    = urgency,
        symptoms   = symptoms or [complaint[:60]],
    )
    display.router(specialists, urgency)
    await loop.run_in_executor(None, tts.wait_until_done)

    # ── Specialist consultations (sequential — one mic, one speaker) ───────────
    reports: list[dict] = []

    for spec_name in specialists:
        display.info(f"Starting {spec_name} consultation...")
        intro = f"I am now connecting you to our {spec_name} specialist."
        tts.speak(intro)
        await loop.run_in_executor(None, tts.wait_until_done)

        try:
            report = await _run_voice_specialist(
                spec_name  = spec_name,
                patient_id = patient_id,
                session_id = session_id,
                complaint  = complaint,
                text_mode  = text_mode,
            )
            reports.append(report)
            display.diagnosis_summary(
                specialty    = spec_name,
                diagnosis    = report.get("diagnosis", ""),
                severity     = report.get("severity", "medium"),
                medications  = report.get("medications", []),
                tests        = report.get("recommended_tests", []),
                follow_up    = report.get("follow_up", ""),
            )
        except Exception as exc:
            display.error(spec_name, str(exc))

    if not reports:
        display.error("VOICE_PIPELINE", "No specialist reports generated — session aborted.")
        tts.stop()
        return {}

    # ── Synthesis ──────────────────────────────────────────────────────────────
    summary = _synthesize_reports(reports, patient)
    display.synthesis(
        lead_role        = summary.get("primary_specialty", "General Medicine"),
        specialist_count = summary.get("specialist_count", 0),
        resource_count   = len(summary.get("all_medications", [])),
    )

    # ── MongoDB save ───────────────────────────────────────────────────────────
    session_doc = {
        "patient_id":           patient_id,
        "session_id":           session_id,
        "urgency":              urgency,
        "initial_complaint":    complaint,
        "symptoms":             symptoms,
        "specialists_consulted":specialists,
        "specialist_reports":   reports,
        "final_summary":        summary,
        "created_at":           datetime.now(timezone.utc).isoformat(),
    }
    try:
        await save_consultation(session_doc)
        display.mongo_save("patient_consultations", patient_id)

        await save_medical_summary(summary)
        display.mongo_save("medical_summaries", patient_id)
    except Exception as exc:
        display.error("MONGO", str(exc))

    # ── Oversight check (calls hospital oversight logic) ──────────────────────
    fraud_flags: list[str] = []
    if len(reports) > 0:
        # Duplicate medication check (collusion indicator across specialists)
        med_counts: dict[str, list[str]] = {}
        for r in reports:
            for m in r.get("medications", []):
                n = m.get("name", "")
                if n:
                    med_counts.setdefault(n, []).append(r.get("specialty", ""))

        for med, claimants in med_counts.items():
            if len(claimants) > 2:
                fraud_flags.append(f"DUPLICATE_PRESCRIPTION: {med} by {claimants}")

    oversight_status = "REJECTED" if fraud_flags else "APPROVED"
    reward = -2.0 * len(fraud_flags) + (1.0 if not fraud_flags else 0.0)
    display.oversight_check(oversight_status, fraud_flags)
    display.decision(oversight_status, reward)

    # ── Final report read aloud ────────────────────────────────────────────────
    display.final_report(
        patient_name      = patient.get("name", ""),
        primary_diagnosis = summary.get("primary_diagnosis", ""),
        all_medications   = summary.get("all_medications", []),
        all_tests         = summary.get("all_tests", []),
        summary           = summary.get("detailed_summary", ""),
    )

    tts.speak("Here is your medical summary. " + summary.get("plain_language", ""))
    await loop.run_in_executor(None, tts.wait_until_done)

    display.info(f"Session {session_id} complete. Goodbye!")
    tts.stop()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pid              = "P1001"
    text_mode        = False
    preset_complaint = ""

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("P") and arg[1:].isdigit():
            pid = arg
        elif arg == "--text":
            text_mode = True
        elif arg == "--complaint" and i + 1 < len(args):
            preset_complaint = args[i + 1]
            i += 1
        i += 1

    asyncio.run(run_voice_session(
        patient_id       = pid,
        text_mode        = text_mode,
        preset_complaint = preset_complaint,
    ))
