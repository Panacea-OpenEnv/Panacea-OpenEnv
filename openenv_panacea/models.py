from pydantic import BaseModel, Field
from typing import Optional, Literal


class Action(BaseModel):
    pass


class Observation(BaseModel):
    done: bool = False
    reward: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class OversightAction(Action):
    """Either a tool_call (gather evidence) or a verdict (terminal decision)."""
    type: Literal["tool_call", "verdict"] = "verdict"
    tool_name: Optional[str] = None
    verdict: str = "REJECTED"
    reasoning: str = ""


class ToolCallAction(OversightAction):
    type: Literal["tool_call", "verdict"] = "tool_call"


class VerdictAction(OversightAction):
    type: Literal["tool_call", "verdict"] = "verdict"


class ClaimDetail(BaseModel):
    claim_id: int = 0
    department: str = ""
    patient_id: str = ""
    requested_resource: str = ""
    claimed_amount: float = 0.0
    protocol: str = ""
    specialist_role: str = ""


class PatientRecord(BaseModel):
    patient_id: str = ""
    name: str = ""
    age: int = 0
    gender: str = ""
    blood_group: str = ""
    vitals: dict = Field(default_factory=dict)
    comorbidities: list[dict] = Field(default_factory=list)
    severity_index: float = 0.0


class OversightObservation(Observation):
    claim: ClaimDetail = Field(default_factory=ClaimDetail)
    patient: Optional[PatientRecord] = None
    department_trust: float = 1.0
    step_number: int = 0
    max_steps: int = 8
    prompt: str = ""
    tools_used: dict[str, bool] = Field(default_factory=dict)
    tool_cost_total: float = 0.0
    last_tool_evidence: str = ""


class PanaceaEpisodeState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    deception_type: str = "none"
    verdict_history: list[str] = Field(default_factory=list)
