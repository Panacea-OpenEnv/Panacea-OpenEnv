"""
Inference Server Client — Remote Oversight Model

Thin HTTP client that calls a remote Colab/HuggingFace endpoint to get
the trained RL model's verdict. Falls back to deterministic rules if
the endpoint is unreachable — production never breaks.

Required .env variables:
  OVERSIGHT_ENDPOINT = https://abc.ngrok.io/generate
  OVERSIGHT_API_KEY  = (optional)

Usage:
  result = await query_oversight_model(prompt)
  # result = {"decision": "REJECTED", "fraud_flags": ["ghost"], "reasoning": "..."}
"""

import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

OVERSIGHT_ENDPOINT = os.getenv("OVERSIGHT_ENDPOINT", "")
OVERSIGHT_API_KEY = os.getenv("OVERSIGHT_API_KEY", "")


def query_oversight_model_sync(prompt: str, timeout: float = 30.0) -> dict:
    """
    Synchronous version for use inside LangGraph nodes (which are plain def, not async).
    Uses httpx synchronously when endpoint is set; falls back to deterministic rules.
    """
    if not OVERSIGHT_ENDPOINT:
        return _deterministic_fallback(prompt)
    try:
        import httpx
        headers = {"Authorization": f"Bearer {OVERSIGHT_API_KEY}"} if OVERSIGHT_API_KEY else {}
        resp = httpx.post(
            OVERSIGHT_ENDPOINT,
            json={"prompt": prompt},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_text = resp.json().get("text", resp.json().get("generated_text", ""))
        return _parse_response(raw_text) if raw_text else _deterministic_fallback(prompt)
    except Exception as e:
        print(f"[INFERENCE] Sync endpoint error ({e}), using deterministic fallback")
        return _deterministic_fallback(prompt)


async def query_oversight_model(prompt: str, timeout: float = 30.0) -> dict:
    """
    POST prompt to remote RL model endpoint.
    Returns parsed {decision, fraud_flags, reasoning}.
    Falls back to deterministic rules on any failure.
    """
    if not OVERSIGHT_ENDPOINT:
        return _deterministic_fallback(prompt)

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            headers = {}
            if OVERSIGHT_API_KEY:
                headers["Authorization"] = f"Bearer {OVERSIGHT_API_KEY}"

            resp = await client.post(
                OVERSIGHT_ENDPOINT,
                json={"prompt": prompt},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()

            body = resp.json()
            raw_text = body.get("text", body.get("generated_text", ""))

            if raw_text:
                return _parse_response(raw_text)
            else:
                return _deterministic_fallback(prompt)

    except Exception as e:
        print(f"[INFERENCE] Endpoint error ({e}), using deterministic fallback")
        return _deterministic_fallback(prompt)


def _parse_response(text: str) -> dict:
    """Extract VERDICT and REASONING from LLM output."""
    decision = "REJECTED"  # Safe default
    reasoning = ""
    fraud_flags = []

    # Try structured VERDICT: line
    verdict_match = re.search(r"VERDICT:\s*(APPROVED|REJECTED|PARTIAL)", text, re.IGNORECASE)
    if verdict_match:
        decision = verdict_match.group(1).upper()

    # Try structured REASONING: line
    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Try to extract JSON if present
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            parsed = json.loads(text[json_start:json_end])
            if isinstance(parsed, dict):
                decision = parsed.get("decision", parsed.get("verdict", decision)).upper()
                fraud_flags = parsed.get("fraud_flags", fraud_flags)
                reasoning = parsed.get("reasoning", reasoning)
    except (json.JSONDecodeError, ValueError):
        pass

    # Detect fraud types from reasoning text
    if not fraud_flags and reasoning:
        lower = reasoning.lower()
        if "ghost" in lower or "no patient" in lower or "not found" in lower:
            fraud_flags.append("ghost")
        if "inflat" in lower or "overcharg" in lower or "excessive" in lower:
            fraud_flags.append("inflation")
        if "mask" in lower or "hidden" in lower or "omit" in lower:
            fraud_flags.append("masking")
        if "collus" in lower or "identical" in lower or "same drug" in lower:
            fraud_flags.append("collusion")

    return {
        "decision": decision,
        "fraud_flags": fraud_flags,
        "reasoning": reasoning,
    }


def _deterministic_fallback(prompt: str) -> dict:
    """
    Rule-based fallback when the RL model endpoint is unavailable.
    Uses simple heuristics that mirror the ground truth signals.
    """
    decision = "APPROVED"
    fraud_flags = []
    reasoning = "Deterministic analysis: "

    prompt_lower = prompt.lower()

    # Ghost detection: no patient record
    if "no patient record found" in prompt_lower or "*** no patient" in prompt_lower:
        decision = "REJECTED"
        fraud_flags.append("ghost")
        reasoning += "Patient ID not found in hospital registry. Ghost patient detected."
        return {"decision": decision, "fraud_flags": fraud_flags, "reasoning": reasoning}

    # Collusion detection: same drug in multiple reports
    import re as _re
    med_sections = _re.findall(r"medications:\s*([^\n]+)", prompt_lower)
    if len(med_sections) >= 2:
        drugs_per_report = []
        for section in med_sections:
            drugs = [d.strip() for d in section.split(",")]
            drugs_per_report.append(set(drugs))

        if len(drugs_per_report) >= 2:
            shared = drugs_per_report[0]
            for ds in drugs_per_report[1:]:
                shared = shared & ds
            if shared:
                decision = "REJECTED"
                fraud_flags.append("collusion")
                reasoning += f"Multiple specialists prescribing identical drugs: {shared}."
                return {"decision": decision, "fraud_flags": fraud_flags, "reasoning": reasoning}

    # Inflation detection: check claimed vs expected range
    amount_match = _re.search(r"claimed amount:\s*\$?([\d,]+\.?\d*)", prompt_lower)
    severity_match = _re.search(r"severity index:\s*([\d.]+)", prompt_lower)

    if amount_match and severity_match:
        try:
            claimed = float(amount_match.group(1).replace(",", ""))
            severity = float(severity_match.group(1))
            # Rough expected: base_cost ~20000 * severity
            expected_high = 25000.0 * severity * 1.5
            if claimed > expected_high:
                decision = "REJECTED"
                fraud_flags.append("inflation")
                reasoning += (
                    f"Claimed ${claimed:,.2f} exceeds expected range for severity "
                    f"{severity:.3f} (max ~${expected_high:,.2f})."
                )
                return {"decision": decision, "fraud_flags": fraud_flags, "reasoning": reasoning}
        except (ValueError, TypeError):
            pass

    reasoning += "All checks passed. Claim appears legitimate."
    return {"decision": decision, "fraud_flags": fraud_flags, "reasoning": reasoning}
