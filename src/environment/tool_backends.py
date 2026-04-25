"""
Mock multi-app tool backends for the Panacea POMDP environment.

Each backend simulates a separate enterprise system with its own:
  - schema (different field shapes / wording)
  - reliability (probability the call succeeds)
  - latency penalty (small reward cost for using slow APIs)

The oversight agent must learn which subset of tools to call to detect
each deception type efficiently — this is the multi-app workflow signal
the theme is asking for.
"""

import random
from typing import Optional

# ── Backend metadata ─────────────────────────────────────────────────────────

TOOL_BACKENDS: dict[str, dict] = {
    "TOOL_REGISTRY": {
        "app": "Government ID Verification API",
        "reliability": 0.97,
        "latency_penalty": -0.05,
        "base_cost": -0.10,
        "description": "Confirms patient existence in national registry.",
    },
    "TOOL_VITALS": {
        "app": "Hospital EMR System",
        "reliability": 0.99,
        "latency_penalty": -0.05,
        "base_cost": -0.10,
        "description": "Returns current vitals and severity index.",
    },
    "TOOL_REPORTS": {
        "app": "Specialist Report Portal",
        "reliability": 0.95,
        "latency_penalty": -0.10,
        "base_cost": -0.15,
        "description": "Pulls all specialist reports filed for the encounter.",
    },
    "TOOL_DRUGS": {
        "app": "National Drug Interaction DB",
        "reliability": 0.90,
        "latency_penalty": -0.10,
        "base_cost": -0.10,
        "description": "Cross-checks prescribed drugs across specialists.",
    },
    "TOOL_BILLING": {
        "app": "Insurance Claims System",
        "reliability": 0.85,
        "latency_penalty": -0.15,
        "base_cost": -0.10,
        "description": "Returns expected cost vs claimed amount and history.",
    },
}

TOOL_NAMES = list(TOOL_BACKENDS.keys())  # stable order — index in obs vector

UNAVAILABLE = "[TOOL UNAVAILABLE — API timeout]"


def tool_cost(tool_name: str) -> float:
    """Total reward penalty for invoking this tool (base + latency)."""
    b = TOOL_BACKENDS[tool_name]
    return b["base_cost"] + b["latency_penalty"]


def _rand() -> random.Random:
    # Episode-local randomness lives in the env; tools just use module random.
    return random


# ── Per-tool view formatters ─────────────────────────────────────────────────
#
# Each formatter takes the raw `episode` dict (from ScenarioGenerator) and
# returns the partial view that THIS app would expose. Different schemas on
# purpose — the agent has to integrate heterogeneous evidence.


def _view_registry(episode: dict) -> str:
    """Government ID Verification API — patient existence check."""
    patient = episode.get("patient")
    pid = episode.get("patient_id", "UNKNOWN")
    if patient is None:
        return (
            f"[REGISTRY] Lookup pid={pid}: NO RECORD FOUND. "
            f"Patient ID does not exist in national registry."
        )
    return (
        f"[REGISTRY] Lookup pid={pid}: VERIFIED. "
        f"name='{patient['name']}', age={patient['age']}, "
        f"gender={patient['gender']}, blood_group={patient['blood_group']}."
    )


def _view_vitals(episode: dict) -> str:
    """Hospital EMR System — vitals and severity index."""
    patient = episode.get("patient")
    if patient is None:
        return "[EMR] No vitals on file for this patient ID."
    v = patient["vitals"]
    return (
        f"[EMR] vitals: HR={v['heart_rate']} BP={v['bp_systolic']}/{v['bp_diastolic']} "
        f"Temp={v['temperature']}C SpO2={v['o2_saturation']}% RR={v['respiratory_rate']} | "
        f"severity_index={patient['severity_index']}"
    )


def _view_reports(episode: dict) -> str:
    """Specialist Report Portal — diagnosis/severity per specialist."""
    reports = episode.get("reports", [])
    if not reports:
        return "[REPORTS] No specialist reports filed for this encounter."
    lines = []
    for i, r in enumerate(reports, 1):
        lines.append(
            f"  #{i} {r['specialty']} ({r['role']}): "
            f"dx='{r['diagnosis']}' severity={r['severity']} "
            f"summary={r['summary'][:120]}"
        )
    # Visible comorbidities for masking scenarios — same logic as prompt formatter
    if episode.get("visible_comorbidities") is not None:
        c = ", ".join(c["condition"] for c in episode["visible_comorbidities"]) or "none"
        lines.append(f"  comorbidities_disclosed: {c}")
    elif episode.get("patient"):
        c = ", ".join(c["condition"] for c in episode["patient"].get("comorbidities", [])) or "none"
        lines.append(f"  comorbidities_disclosed: {c}")
    return "[REPORTS]\n" + "\n".join(lines)


def _view_drugs(episode: dict) -> str:
    """National Drug Interaction DB — cross-specialist drug overlap."""
    reports = episode.get("reports", [])
    if not reports:
        return "[DRUGDB] No prescriptions linked to this encounter."
    drug_to_specs: dict[str, list[str]] = {}
    for r in reports:
        for m in r.get("medications", []):
            drug_to_specs.setdefault(m["name"], []).append(r["specialty"])
    lines = []
    for drug, specs in drug_to_specs.items():
        flag = " <DUPLICATE-PRESCRIBER>" if len(specs) > 1 else ""
        lines.append(f"  {drug}: prescribed_by={specs}{flag}")
    return "[DRUGDB]\n" + "\n".join(lines)


def _view_billing(episode: dict) -> str:
    """Insurance Claims System — expected vs claimed amount."""
    claimed = episode.get("claimed_amount", 0.0)
    expected = episode.get("expected_cost", 0.0)
    ratio = (claimed / expected) if expected else float("inf")
    flag = ""
    if expected == 0:
        flag = " <NO-EXPECTED-COST-ON-FILE>"
    elif ratio >= 1.5:
        flag = f" <RATIO={ratio:.2f}x EXPECTED>"
    return (
        f"[BILLING] claimed=${claimed:,.2f} expected=${expected:,.2f}{flag}"
    )


_VIEWERS = {
    "TOOL_REGISTRY": _view_registry,
    "TOOL_VITALS": _view_vitals,
    "TOOL_REPORTS": _view_reports,
    "TOOL_DRUGS": _view_drugs,
    "TOOL_BILLING": _view_billing,
}


def call_tool(tool_name: str, episode: dict, rng: Optional[random.Random] = None) -> str:
    """
    Invoke a backend against the current episode.

    Returns a string the agent sees as accumulated context. On reliability
    failure, returns UNAVAILABLE — the agent must cope with partial info.
    """
    if tool_name not in TOOL_BACKENDS:
        raise ValueError(f"Unknown tool: {tool_name}")

    r = rng if rng is not None else random
    if r.random() > TOOL_BACKENDS[tool_name]["reliability"]:
        app = TOOL_BACKENDS[tool_name]["app"]
        return f"[{app}] {UNAVAILABLE}"

    return _VIEWERS[tool_name](episode)
