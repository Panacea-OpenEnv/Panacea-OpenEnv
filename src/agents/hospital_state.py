from typing import TypedDict, Annotated
import operator


class HospitalState(TypedDict):
    #  Patient context 
    patient_id: str
    patient_name: str
    age: int
    symptoms: list[str]
    conditions: list[str]
    urgency: str            # "critical" | "high" | "medium" | "low"
    patient_db_record: dict

    #  Routing & dispatch 
    # All list fields use operator.add so parallel Send nodes don't clobber each other
    active_specialists: Annotated[list[str], operator.add]
    completed_specialists: Annotated[list[str], operator.add]
    pending_consultations: Annotated[list[str], operator.add]
    consultation_round: int
    current_specialist: str     # injected by Send; tells run_specialist who it is

    #  Cross-agent communication (append-only) 
    specialist_reports: Annotated[list[dict], operator.add]
    consultation_results: Annotated[list[dict], operator.add]
    inter_agent_messages: Annotated[list[dict], operator.add]
    resource_requests: Annotated[list[dict], operator.add]

    #  Treatment synthesis 
    treatment_plan: dict

    #  Oversight 
    oversight_decision: str     # "APPROVED" | "REJECTED" | "PARTIAL" | "PENDING"
    fraud_flags: Annotated[list[str], operator.add]

    #  Audit trail 
    # step_count uses operator.add so parallel Send nodes can each contribute +1
    reasoning: Annotated[list[str], operator.add]
    reward: float
    step_count: Annotated[int, operator.add]


def initial_hospital_state(
    patient_id: str = "P1001",
    symptoms: list[str] | None = None,
    conditions: list[str] | None = None,
    urgency: str = "medium",
) -> HospitalState:
    return HospitalState(
        patient_id=patient_id,
        patient_name="",
        age=0,
        symptoms=symptoms or [],
        conditions=conditions or [],
        urgency=urgency,
        patient_db_record={},
        active_specialists=[],
        completed_specialists=[],
        pending_consultations=[],
        consultation_round=0,
        current_specialist="",
        specialist_reports=[],
        consultation_results=[],
        inter_agent_messages=[],
        resource_requests=[],
        treatment_plan={},
        oversight_decision="PENDING",
        fraud_flags=[],
        reasoning=[],
        reward=0.0,
        step_count=0,
    )
