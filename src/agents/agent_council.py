"""
Agent Council — Internal Doctor-to-Doctor Communication Layer

After detecting the primary specialist from the patient's first complaint,
the primary doctor consults peer agents internally. All communication is
shown on the terminal. The patient is NOT involved in this phase.

Flow:
  1. detect_primary_specialist(complaint) → primary specialty name
  2. run_council(primary, complaint) → list of targeted questions to ask patient
     - Primary agent broadcasts to consultation_partners
     - Each peer responds with clinical observations
     - Council is summarised into patient questions
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from .specialists.registry import SPECIALISTS

load_dotenv()

_client  = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")
console  = Console()

#  Color map per specialty 
_COLORS = [
    "cyan", "green", "yellow", "magenta", "blue",
    "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
]

def _color(specialty: str) -> str:
    names = list(SPECIALISTS.keys())
    idx   = names.index(specialty) % len(_COLORS) if specialty in names else 0
    return _COLORS[idx]


#  — Detect primary specialist from raw complaint 

_DETECT_PROMPT = f"""\
You are a medical triage AI. Given the patient's first complaint, output ONLY JSON:
{{
  "primary_specialist": "<one specialty name from the list>",
  "confidence": "high|medium|low",
  "reason": "one sentence"
}}

VALID SPECIALTIES: {", ".join(SPECIALISTS.keys())}

Rules:
- Choose the single most relevant specialty for the chief complaint.
- If unclear, choose "General Medicine".
"""

async def detect_primary_specialist(complaint: str) -> dict:
    """
    Immediately detect the primary specialist from the patient's first message.
    Returns dict: {primary_specialist, confidence, reason}
    """
    try:
        resp = await _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _DETECT_PROMPT},
                {"role": "user",   "content": complaint},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw   = resp.choices[0].message.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(raw[start:end])
            if result.get("primary_specialist") in SPECIALISTS:
                return result
    except Exception:
        pass
    return {"primary_specialist": "General Medicine", "confidence": "low", "reason": "Fallback"}


#  — Peer consultation (agent-to-agent, shown on terminal) 

async def _peer_consult(
    primary: str,
    peer: str,
    complaint: str,
    primary_assessment: str,
) -> str:
    """
    Primary specialist sends a clinical question to a peer.
    Returns peer's response text.
    """
    peer_info = SPECIALISTS.get(peer, {})
    prompt = f"""\
You are a {peer_info.get('role', peer)} consultant at Panacea Hospital.
A {SPECIALISTS[primary]['role']} has asked for your clinical input on a patient.

Patient complaint: {complaint}
Primary specialist's initial assessment: {primary_assessment}

Respond in 2-3 sentences with your specific clinical observations, red flags to check,
or tests you would recommend from your specialty perspective.
Be direct and clinical. No markdown."""

    try:
        resp = await _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "No specific concerns from this specialty at this time."


async def _primary_assessment(primary: str, complaint: str) -> str:
    """Primary specialist forms an initial clinical picture from the complaint."""
    spec_info = SPECIALISTS[primary]
    prompt = f"""\
You are a {spec_info['role']} at Panacea Hospital.
A patient has just arrived with this complaint: "{complaint}"

In 2-3 sentences, state your initial clinical impression and what additional
information you need from the patient. Be direct and clinical. No markdown."""

    try:
        resp = await _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Initial assessment pending for complaint: {complaint}"


async def _synthesize_questions(
    primary: str,
    complaint: str,
    primary_assessment: str,
    peer_opinions: list[dict],
) -> list[str]:
    """
    After all agents have spoken, generate 3-5 targeted patient questions.
    These are what the doctor will actually ask the patient.
    """
    opinions_text = "\n".join(
        f"- {p['peer']} ({SPECIALISTS[p['peer']]['role']}): {p['opinion']}"
        for p in peer_opinions
    )

    prompt = f"""\
You are a {SPECIALISTS[primary]['role']} who has just consulted with colleagues.

Patient complaint: {complaint}
Your initial assessment: {primary_assessment}
Colleague input:
{opinions_text}

Based on all this, generate exactly 3 to 5 targeted clinical questions to ask the patient.
Each question should be simple, clear, one sentence. Output ONLY a JSON array:
["question 1", "question 2", "question 3"]"""

    try:
        resp = await _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        raw   = resp.choices[0].message.content.strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            questions = json.loads(raw[start:end])
            if isinstance(questions, list) and questions:
                return questions[:5]
    except Exception:
        pass
    return [
        "How long have you had this symptom?",
        "On a scale of 1 to 10, how severe is the pain or discomfort?",
        "Does anything make it better or worse?",
    ]


#  Main council runner 

async def run_council(primary: str, complaint: str) -> dict:
    """
    Run the full internal agent council:
      1. Primary specialist forms initial assessment (shown on terminal)
      2. Consults up to 3 peer specialists (shown on terminal)
      3. Synthesizes into patient questions

    Returns:
      {
        "primary": str,
        "peer_opinions": [{"peer": str, "opinion": str}],
        "patient_questions": [str],
        "primary_assessment": str,
      }
    """
    spec_info   = SPECIALISTS.get(primary, {})
    primary_role = spec_info.get("role", primary)
    primary_color = _color(primary)
    partners    = spec_info.get("consultation_partners", [])[:3]

    # Header
    console.print()
    console.print(Rule(f"[bold white] DOCTOR COUNCIL — Internal Consultation [/]", style="white"))
    console.print(f"  [dim]Patient complaint:[/] {complaint}\n")

    # — Primary assessment
    console.print(f"  [{primary_color}][{primary_role}][/] forming initial assessment...")
    primary_assessment = await _primary_assessment(primary, complaint)

    console.print(
        Panel(
            Text(primary_assessment, style="white"),
            title=f"[bold {primary_color}]🩺 {primary_role} — Initial Assessment[/]",
            border_style=primary_color,
            padding=(0, 1),
        )
    )

    # — Peer consultations
    peer_opinions: list[dict] = []

    for peer in partners:
        if peer not in SPECIALISTS:
            continue

        peer_role  = SPECIALISTS[peer].get("role", peer)
        peer_color = _color(peer)

        # Show the outgoing message
        console.print(
            f"\n  [{primary_color}][{primary_role}][/] "
            f"[dim]→[/] "
            f"[{peer_color}][{peer_role}][/]  "
            f"[dim]requesting clinical input...[/]"
        )

        opinion = await _peer_consult(primary, peer, complaint, primary_assessment)
        peer_opinions.append({"peer": peer, "opinion": opinion})

        # Show peer response
        console.print(
            Panel(
                Text(opinion, style="white"),
                title=f"[bold {peer_color}]{peer_role}[/]  [dim]→  {primary_role}[/]",
                border_style=peer_color,
                padding=(0, 1),
            )
        )

        await asyncio.sleep(0.1)   # small pause for readability

    # — Synthesize patient questions
    console.print(f"\n  [{primary_color}][{primary_role}][/] [dim]synthesizing patient questions from council...[/]\n")
    patient_questions = await _synthesize_questions(
        primary, complaint, primary_assessment, peer_opinions
    )

    console.print(Rule(style="dim"))
    console.print(f"  [bold white]Council complete.[/] {len(peer_opinions)} peers consulted.\n")

    return {
        "primary":            primary,
        "primary_role":       primary_role,
        "primary_assessment": primary_assessment,
        "peer_opinions":      peer_opinions,
        "patient_questions":  patient_questions,
    }
