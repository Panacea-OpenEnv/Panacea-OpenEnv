"""
Voice Pipeline — Panacea Patient Intake (Full Proper Flow)

Flow:
  1. Greet patient (TTS)
  2. Patient states complaint (STT)
  3. Intake Nurse interviews patient (3-5 clarifying Q&A turns)
  4. Smart Router assigns specialist(s) based on structured intake summary
  5. Per specialist: multi-turn voice consultation
       - GPT-4o streams tokens → TTS speaks sentence-by-sentence
       - Patient speaks → STT transcribes → injected into specialist
       - If specialist requests consult, bridge runs cross-specialist Q&A
  6. Synthesize all specialist reports → rich prescription
  7. Save session to MongoDB
  8. TTS reads final prescription aloud

Usage:
  python -m src.voice.voice_pipeline P1001
  python -m src.voice.voice_pipeline P1001 --text
"""

import asyncio
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

from .stt import stt
from .tts import tts
from ..agents.smart_router import route_complaint
from ..agents.specialist_gpt import run_specialist_consultation, inject_patient_reply
from ..agents.consult_bridge import detect_consult_request, run_consult
from ..agents.agent_council import detect_primary_specialist, run_council
from ..database.mongo_client import get_patient, save_consultation, save_medical_summary
from ..environment.reward import compute_reward
from ..utils.terminal_display import display

load_dotenv()

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Shared patient reply for consult bridge
# ─────────────────────────────────────────────────────────────────────────────

_consult_reply_queue: asyncio.Queue = asyncio.Queue()

async def _get_consult_patient_reply() -> str:
    return await _consult_reply_queue.get()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Single-specialist voice consultation with Phase 3 consult bridge
# ─────────────────────────────────────────────────────────────────────────────

async def _run_voice_specialist(
    spec_name: str,
    patient_id: str,
    patient: dict,
    session_id: str,
    complaint: str,
    text_mode: bool,
    loop: asyncio.AbstractEventLoop,
) -> dict:
    """
    Multi-turn specialist consultation with live cross-specialist consult detection.
    """
    turn_signal: asyncio.Queue = asyncio.Queue()
    response_buffer = []
    json_detected   = False
    consults_done: set[str] = set()

    def on_token(token: str):
        nonlocal json_detected
        response_buffer.append(token)
        so_far = "".join(response_buffer).lstrip()
        if so_far.startswith("{"):
            json_detected = True
        if not json_detected:
            tts.stream_token(token)

    def on_turn_done(turn_info: dict):
        nonlocal json_detected
        if not json_detected:
            tts.flush_buffer()
        turn_signal.put_nowait({**turn_info, "buffer": "".join(response_buffer)})
        response_buffer.clear()
        json_detected = False

    spec_task = asyncio.create_task(
        run_specialist_consultation(
            spec_name=spec_name,
            patient_id=patient_id,
            session_id=f"{session_id}_{spec_name.replace(' ', '_')}",
            initial_complaint=complaint,
            on_token=on_token,
            on_turn_complete=on_turn_done,
            voice_mode=True,
        )
    )

    async def inject_loop():
        while not spec_task.done():
            try:
                turn_info = await asyncio.wait_for(turn_signal.get(), timeout=120.0)
            except asyncio.TimeoutError:
                display.error(spec_name, "Turn timeout")
                break

            await loop.run_in_executor(None, tts.wait_until_done)

            if spec_task.done():
                break

            # If the specialist just delivered the final JSON assessment, stop here.
            # spec_task.done() may still be False (awaiting MongoDB save), so we check
            # the buffer directly instead of relying on task completion timing.
            full_text = turn_info.get("buffer", turn_info.get("response", ""))
            if full_text.lstrip().startswith("{"):
                break

            # Phase 3: Check if specialist wants a cross-consult
            consult_req = await detect_consult_request(full_text)
            if consult_req and consult_req.get("consult_specialty") not in consults_done:
                consult_spec = consult_req["consult_specialty"]
                reason       = consult_req.get("reason", "second opinion requested")
                consults_done.add(consult_spec)

                display.consult_bridge(spec_name, consult_spec, reason)
                tts.speak(
                    f"I would like to bring in our {consult_spec} specialist "
                    f"for a quick second opinion. They will ask you one question."
                )
                await loop.run_in_executor(None, tts.wait_until_done)

                # Run the consult (1-2 turns with patient)
                async def get_consult_reply():
                    tts.speak("Please go ahead.")
                    await loop.run_in_executor(None, tts.wait_until_done)
                    if text_mode:
                        print("\nYour reply: ", end="", flush=True)
                        return input().strip() or "I am not sure."
                    else:
                        display.patient_listening()
                        return await loop.run_in_executor(None, stt.record_until_silence)

                consult_opinion = await run_consult(
                    consult_specialty=consult_spec,
                    primary_specialty=spec_name,
                    patient=patient,
                    primary_findings=full_text[:500],
                    on_token=on_token,
                    voice_mode=True,
                    get_patient_reply=get_consult_reply,
                )

                await loop.run_in_executor(None, tts.wait_until_done)
                display.consult_result(consult_spec, consult_opinion.get("opinion", ""))

                tts.speak(f"Thank you. The {consult_spec} specialist has shared their findings. Let me continue with your assessment.")
                await loop.run_in_executor(None, tts.wait_until_done)

                # Inject consult opinion + ask primary specialist to finalize
                opinion_text = consult_opinion.get("opinion", "")
                extra_meds   = consult_opinion.get("additional_medications", [])
                extra_tests  = consult_opinion.get("additional_tests", [])

                consult_note = (
                    f"[CONSULT FROM {consult_spec}: {opinion_text}"
                    + (f" | Recommended additional medications: {[m.get('name','') for m in extra_meds]}" if extra_meds else "")
                    + (f" | Recommended additional tests: {extra_tests}" if extra_tests else "")
                    + "] Please incorporate this into your final assessment now."
                )
                await inject_patient_reply(consult_note)
                continue

            # Normal turn — get patient reply
            tts.speak("Go ahead, I am listening.")
            await loop.run_in_executor(None, tts.wait_until_done)

            if text_mode:
                print(f"\n[{spec_name}] Your reply: ", end="", flush=True)
                reply = input().strip()
            else:
                # Flush mic buffer so TTS output isn't transcribed as patient speech
                await loop.run_in_executor(None, stt.flush_input_buffer)
                display.patient_listening()
                reply = await loop.run_in_executor(None, stt.record_until_silence)

            reply = reply or "I am not sure, could you explain more?"
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

    tts.flush_buffer()

    # Attach any consults that happened
    if consults_done:
        report["consults_requested"] = list(consults_done)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Rich prescription synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _synthesize_reports(reports: list[dict], patient: dict, routing: dict) -> dict:
    if not reports:
        return {}

    lead = max(reports, key=lambda r: _SEVERITY_RANK.get(r.get("severity", "unknown"), 0))

    all_meds:  list[dict] = []
    all_tests: list[str]  = []
    seen_meds:  set[str]  = set()
    seen_tests: set[str]  = set()
    drug_conflicts: list[str] = []

    for r in reports:
        for m in r.get("medications", []):
            name = m.get("name", "")
            if name and name not in seen_meds:
                seen_meds.add(name)
                all_meds.append(m)
            elif name in seen_meds:
                drug_conflicts.append(f"{name} prescribed by multiple specialists")

    for r in reports:
        for t in r.get("recommended_tests", []):
            if t and t not in seen_tests:
                seen_tests.add(t)
                all_tests.append(t)

    # Prioritize: urgent tests first
    urgent_keywords = ["ECG", "CT", "MRI", "biopsy", "culture", "troponin", "CBC"]
    all_tests.sort(key=lambda t: any(kw.lower() in t.lower() for kw in urgent_keywords), reverse=True)

    summaries = [r.get("summary", "") for r in reports if r.get("summary")]
    combined  = " ".join(summaries)

    plain = (
        f"Based on the assessments of our specialist team, your primary condition is "
        f"{lead.get('diagnosis', 'under evaluation')}. "
        + (
            f"You have been prescribed {', '.join(m['name'] for m in all_meds[:3])}. "
            if all_meds else ""
        )
        + (
            f"Please get the following tests done: {', '.join(all_tests[:3])}. "
            if all_tests else ""
        )
        + "Please follow your follow-up instructions carefully and rest well."
    )

    follow_up = lead.get("follow_up", "Follow up with your doctor in 7 days.")

    return {
        "patient_id":           patient.get("patient_id", ""),
        "patient_name":         patient.get("name", ""),
        "primary_diagnosis":    lead.get("diagnosis", ""),
        "primary_specialty":    lead.get("specialty", ""),
        "severity":             lead.get("severity", "medium"),
        "all_medications":      all_meds,
        "all_tests":            all_tests,
        "follow_up":            follow_up,
        "specialist_count":     len(reports),
        "specialists_involved": [r.get("specialty", "") for r in reports],
        "drug_conflicts":       drug_conflicts,
        "routing_reason":       routing.get("routing_reason", ""),
        "detailed_summary":     combined,
        "plain_language":       plain,
        "created_at":           datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main session runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_voice_session(
    patient_id: str = "P1001",
    text_mode:  bool = False,
    preset_complaint: str = "",
) -> dict:
    session_id = str(uuid.uuid4())[:12]
    loop       = asyncio.get_event_loop()

    if not text_mode:
        stt.load()
    tts.start()

    patient_from_db = await get_patient(patient_id)
    patient = patient_from_db or {
        "patient_id": patient_id,
        "name": "Patient",
    }
    patient_name = patient.get("name", "Patient")

    display.intake_nurse_header(patient_name)
    tts.speak(
        f"Welcome to Panacea Hospital, {patient_name}. "
        f"I am Priya, your intake nurse. "
        f"Please tell me what brings you in today."
    )
    await loop.run_in_executor(None, tts.wait_until_done)

    # ── Get initial complaint ──────────────────────────────────────────────────
    if preset_complaint:
        complaint = preset_complaint
        text_mode = True
    elif text_mode:
        print("\nDescribe your symptoms: ", end="", flush=True)
        complaint = input().strip()
    else:
        display.patient_listening()
        complaint = await loop.run_in_executor(None, stt.record_until_silence)

    complaint = complaint or "I have a general health concern."
    display.patient_speech(complaint)

    # ── Phase 1: Immediately detect primary specialist from first complaint ─────
    display.info("Detecting primary specialist from complaint...")
    detection = await detect_primary_specialist(complaint)
    primary   = detection["primary_specialist"]
    display.info(
        f"Primary specialist detected: {primary} "
        f"(confidence: {detection['confidence']}) — {detection['reason']}"
    )

    # ── Phase 2: Internal agent council (doctor-to-doctor, shown on terminal) ───
    # Doctors consult each other BEFORE asking the patient anything further.
    # Patient is NOT involved here — this is purely internal clinical reasoning.
    display.info("Launching internal agent council...")
    council = await run_council(primary, complaint)

    # ── Phase 3: Doctor interviews patient using council-generated questions ────
    # TTS speaks each question; patient answers via voice or text.
    tts.speak(
        f"Thank you for telling me that. I am Doctor {council['primary_role']}. "
        f"I have a few targeted questions for you."
    )
    await loop.run_in_executor(None, tts.wait_until_done)

    patient_answers: list[dict] = []
    for i, question in enumerate(council["patient_questions"], 1):
        display.nurse_question(f"[Dr. {council['primary_role']}] {question}")
        tts.speak(question)
        # wait for TTS to finish before capturing patient reply
        await loop.run_in_executor(None, tts.wait_until_done)

        if text_mode:
            print(f"\nYour answer: ", end="", flush=True)
            answer = input().strip()
        else:
            await loop.run_in_executor(None, stt.flush_input_buffer)
            display.patient_listening()
            answer = await loop.run_in_executor(None, stt.record_until_silence)

        answer = answer or "I am not sure."
        display.patient_speech(answer)
        patient_answers.append({"question": question, "answer": answer})

    # Build structured intake from council + patient answers
    # Include all patient answers so the router sees the real symptoms, not just the initial complaint
    qa_text = "\n".join(f"Q: {a['question']}\nA: {a['answer']}" for a in patient_answers)
    associated_symptoms = [a["answer"] for a in patient_answers if a.get("answer")]
    enriched_complaint = complaint
    if associated_symptoms:
        enriched_complaint = f"{complaint} Additional symptoms reported: {'; '.join(associated_symptoms)}"

    structured = {
        "chief_complaint": enriched_complaint,
        "council_assessment": council["primary_assessment"],
        "peer_opinions": [p["opinion"] for p in council["peer_opinions"]],
        "patient_qa": qa_text,
        "severity": "moderate",
        "severity_signals": [],
        "associated_symptoms": associated_symptoms,
        "modifiers": {},
        "suspected_systems": [],   # Let smart router decide after seeing all answers
        "duration": "as described by patient",
    }

    # ── Phase 4: Smart routing (primary already known, may add secondary) ──────
    tts.speak("Thank you. Let me now connect you with the specialist team.")
    routing = await route_complaint(structured)

    # Use router's decision; only fall back to initial primary if router returned nothing
    specialists_raw: list[str] = routing.get("specialists", [])
    if not specialists_raw:
        specialists_raw = [primary]
    specialists: list[str] = specialists_raw[:3]
    urgency:     str       = routing.get("urgency", "medium")
    reason:      str       = routing.get("routing_reason", "")

    display.session_header(
        patient_id=patient_id,
        session_id=session_id,
        urgency=urgency,
        symptoms=[complaint[:60]],
    )
    display.router(specialists, urgency)
    display.info(f"Routing reason: {reason}")
    await loop.run_in_executor(None, tts.wait_until_done)

    # ── Build clinical handoff note for specialists ───────────────────────────
    # Specialists receive everything gathered so far so they don't re-ask
    # questions the patient already answered during the intake phase.
    handoff_lines = [
        f"PATIENT COMPLAINT: {complaint}",
    ]
    if patient_answers:
        handoff_lines.append("INTAKE Q&A (already collected — do not re-ask these):")
        for a in patient_answers:
            handoff_lines.append(f"  Q: {a['question']}")
            handoff_lines.append(f"  A: {a['answer']}")
    if council.get("primary_assessment"):
        handoff_lines.append(f"CLINICAL PRE-ASSESSMENT: {council['primary_assessment']}")
    if council.get("peer_opinions"):
        opinions = "; ".join(p.get("opinion", "")[:120] for p in council["peer_opinions"])
        handoff_lines.append(f"PEER SPECIALIST INPUT: {opinions}")
    handoff_lines.append(
        "Based on the above context, proceed directly to your specialist assessment. "
        "Do not repeat questions already answered above."
    )
    clinical_handoff = "\n".join(handoff_lines)

    # ── Phase 3: Specialist consultations ─────────────────────────────────────
    reports: list[dict] = []

    for spec_name in specialists:
        display.info(f"Starting {spec_name} consultation...")
        tts.speak(f"I am now connecting you to our {spec_name} specialist.")
        await loop.run_in_executor(None, tts.wait_until_done)

        try:
            report = await _run_voice_specialist(
                spec_name=spec_name,
                patient_id=patient_id,
                patient=patient,
                session_id=session_id,
                complaint=clinical_handoff,
                text_mode=text_mode,
                loop=loop,
            )
            reports.append(report)

            display.diagnosis_summary(
                specialty=spec_name,
                diagnosis=report.get("diagnosis", ""),
                severity=report.get("severity", "medium"),
                medications=report.get("medications", []),
                tests=report.get("recommended_tests", []),
                follow_up=report.get("follow_up", ""),
            )

            tts.speak(
                f"The {spec_name} specialist has completed their assessment. "
                f"Diagnosis: {report.get('diagnosis', 'Inconclusive')}."
            )
            await loop.run_in_executor(None, tts.wait_until_done)

        except Exception as exc:
            display.error(spec_name, str(exc))

    if not reports:
        display.error("VOICE_PIPELINE", "No specialist reports generated.")
        tts.stop()
        return {}

    # ── Phase 4: Rich prescription synthesis ──────────────────────────────────
    summary = _synthesize_reports(reports, patient, routing)

    display.synthesis(
        lead_role=summary.get("primary_specialty", "General Medicine"),
        specialist_count=summary.get("specialist_count", 0),
        resource_count=len(summary.get("all_medications", [])),
    )

    # Drug conflict warning
    conflicts = summary.get("drug_conflicts", [])
    if conflicts:
        display.error("PRESCRIPTION", f"Drug conflicts detected: {conflicts}")

    # Phase 4: Rich prescription display
    display.prescription(
        patient_name=patient_name,
        primary_diagnosis=summary.get("primary_diagnosis", ""),
        severity=summary.get("severity", "medium"),
        all_medications=summary.get("all_medications", []),
        all_tests=summary.get("all_tests", []),
        follow_up=summary.get("follow_up", ""),
        specialists_involved=summary.get("specialists_involved", []),
        summary=summary.get("detailed_summary", ""),
        routing_reason=summary.get("routing_reason", ""),
    )

    # ── Save to MongoDB ────────────────────────────────────────────────────────
    session_doc = {
        "patient_id":           patient_id,
        "session_id":           session_id,
        "urgency":              urgency,
        "initial_complaint":    complaint,
        "structured_intake":    structured,
        "routing":              routing,
        "specialists_consulted":specialists,
        "specialist_reports":   reports,
        "final_summary":        summary,
        "created_at":           datetime.now(timezone.utc).isoformat(),
    }

    try:
        await save_consultation(session_doc)
        display.mongo_save("patient_consultations", patient_id)
    except Exception as exc:
        display.error("MONGO", f"Failed to save consultation: {exc}")

    try:
        await save_medical_summary(summary)
        display.mongo_save("medical_summaries", patient_id)
    except Exception as exc:
        display.error("MONGO", f"Failed to save summary: {exc}")

    # ── Oversight verification (MongoDB-aware, no backend dependency) ────────────
    from ..agents.oversight_core import verify_claim

    verify_result = verify_claim(
        patient_id=patient_id,
        reports=reports,
        resource_requests=[],
        patient_from_db=patient_from_db,
        drug_conflicts=conflicts,
    )

    fraud_flags = verify_result["fraud_flags"]
    oversight_status = verify_result["decision"]
    reasoning = verify_result["reasoning"]

    reward = compute_reward(
        verdict=oversight_status,
        expected_verdict="APPROVED",
        deception_type="none",
        reasoning=reasoning,
    )

    display.oversight_check(oversight_status, fraud_flags)
    display.decision(oversight_status, reward)

    # ── Speak final summary ────────────────────────────────────────────────────
    tts.speak("Here is your medical summary. " + summary.get("plain_language", ""))
    await loop.run_in_executor(None, tts.wait_until_done)

    display.info(f"Session {session_id} complete.")
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
        patient_id=pid,
        text_mode=text_mode,
        preset_complaint=preset_complaint,
    ))
