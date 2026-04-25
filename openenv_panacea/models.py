"""
Pydantic models for the Panacea OpenEnv environment.

OversightAction:     The oversight agent's verdict on a resource claim.
OversightObservation: What the agent sees -- the claim + patient context.
                      Extends openenv Observation base (includes done, reward, metadata).
"""

from pydantic import BaseModel, Field
from typing import Optional

from openenv.core import Observation, Action


class OversightAction(Action):
    """The LLM oversight agent's output for each step."""
    verdict: str = Field(
        default="REJECTED",
        description="APPROVED or REJECTED",
    )
    reasoning: str = Field(
        default="",
        description="Free-text explanation for the verdict",
    )


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
    
    Inherits from OpenEnv Observation which includes:
      - done: bool (whether episode is over)
      - reward: float | None (reward from last action)
      - metadata: dict (additional info)
    """
    claim: ClaimDetail = Field(default_factory=ClaimDetail)
    patient: Optional[PatientRecord] = None
    department_trust: float = 1.0
    step_number: int = 0
    max_steps: int = 1
    prompt: str = ""  # Pre-formatted text prompt for LLM consumption


class PanaceaEpisodeState(BaseModel):
    """Metadata tracked across the episode."""
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    deception_type: str = "none"
    verdict_history: list[str] = Field(default_factory=list)
