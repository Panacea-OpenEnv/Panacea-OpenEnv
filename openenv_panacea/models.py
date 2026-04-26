"""
Pydantic models for the Panacea OpenEnv environment.

The POMDP environment exposes a unified `OversightAction` with two modes:

  type="tool_call"  → invoke an enterprise API to gather evidence
                       (TOOL_REGISTRY, TOOL_VITALS, TOOL_REPORTS,
                        TOOL_DRUGS, TOOL_BILLING).
  type="verdict"    → terminal APPROVED / REJECTED decision.

Backward compat: passing only {verdict, reasoning} still works — the
server will treat it as a verdict action.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class Action(BaseModel):
    """Base action class."""


class Observation(BaseModel):
    """Base observation class."""
    done: bool = False
    reward: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


# ── Unified action with optional discriminator ──────────────────────────────

class OversightAction(Action):
    """Unified action — the agent picks a tool to call OR submits a verdict.

    Field semantics:
      type      — "tool_call" or "verdict"  (default: verdict for back-compat)
      tool_name — required when type="tool_call"
      verdict   — required when type="verdict" ("APPROVED" / "REJECTED")
      reasoning — free-text explanation (used at verdict time)
    """
    type: Literal["tool_call", "verdict"] = Field(
        default="verdict",
        description="Whether this is an evidence-gathering tool call or a final verdict",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="One of TOOL_REGISTRY/TOOL_VITALS/TOOL_REPORTS/TOOL_DRUGS/TOOL_BILLING",
    )
    verdict: str = Field(
        default="REJECTED",
        description="APPROVED or REJECTED (only used when type='verdict')",
    )
    reasoning: str = Field(
        default="",
        description="Free-text explanation for the verdict",
    )


# Convenience aliases — make the two intents discoverable in IDEs
class ToolCallAction(OversightAction):
    """Helper subclass: always type='tool_call'. Pass tool_name."""
    type: Literal["tool_call", "verdict"] = "tool_call"


class VerdictAction(OversightAction):
    """Helper subclass: always type='verdict'. Pass verdict + reasoning."""
    type: Literal["tool_call", "verdict"] = "verdict"


class ClaimDetail(BaseModel):
    """A single resource claim from a specialist sub-agent."""
    claim_id: int = 0
    department: str = ""
    patient_id: str = ""
    requested_resource: str = ""
    claimed_amount: float = 0.0
    protocol: str = ""
    specialist_role: str = ""


class PatientRecord(BaseModel):
    """Patient context visible to the oversight agent."""
    patient_id: str = ""
    name: str = ""
    age: int = 0
    gender: str = ""
    blood_group: str = ""
    vitals: dict = Field(default_factory=dict)
    comorbidities: list[dict] = Field(default_factory=list)
    severity_index: float = 0.0


class OversightObservation(Observation):
    """Everything the oversight agent can see at each step.

    For the POMDP environment:
      - prompt grows as tool calls accumulate evidence
      - tools_used[name] flips to True after each invocation
      - tool_cost_total tracks the running reward debit from API calls
    """
    claim: ClaimDetail = Field(default_factory=ClaimDetail)
    patient: Optional[PatientRecord] = None
    department_trust: float = 1.0
    step_number: int = 0
    max_steps: int = 8
    prompt: str = ""  # Accumulated context for LLM consumption
    tools_used: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-tool boolean: True once invoked at least once this episode",
    )
    tool_cost_total: float = 0.0
    last_tool_evidence: str = Field(
        default="",
        description="Just the evidence string from the most recent tool call, if any",
    )


class PanaceaEpisodeState(BaseModel):
    """Metadata tracked across the episode."""
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    deception_type: str = "none"
    verdict_history: list[str] = Field(default_factory=list)
