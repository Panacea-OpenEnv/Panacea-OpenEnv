"""
Panacea OpenEnv Environment (Server-Side).

Implements the OpenEnv Environment base class with step(), reset(), state().
Each episode presents one adversarial hospital resource claim. The LLM
oversight agent must approve or reject it.
"""

import uuid
from openenv.core import Environment

from ..models import OversightAction, OversightObservation, ClaimDetail, PatientRecord
from ..scenario_generator import generate_scenario
from ..reward import compute_reward


class PanaceaEnvironment(Environment):
    """
    Adversarial hospital oversight environment.

    Per episode:
      1. reset() generates a new claim scenario (with or without deception)
      2. step(action) receives the agent's APPROVED/REJECTED verdict and scores it
      3. state() returns episode metadata
    """

    def __init__(self):
        super().__init__()
        self._episode_id: str = ""
        self._step_count: int = 0
        self._scenario: dict = {}
        self._total_reward: float = 0.0
        self._done: bool = False
        self._verdict_history: list[str] = []

    def reset(self, seed=None, episode_id=None, **kwargs) -> OversightObservation:
        """Generate a new adversarial claim scenario."""
        self._episode_id = episode_id or str(uuid.uuid4())[:12]
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._verdict_history = []

        self._scenario = generate_scenario(seed=seed)

        return self._build_observation(done=False, reward=None)

    def step(self, action: OversightAction, timeout_s=None, **kwargs) -> OversightObservation:
        """
        Receive the agent's verdict and score it.
        """
        self._step_count += 1

        # Parse verdict (handle free-form LLM output)
        verdict = self._parse_verdict(
            action.verdict if hasattr(action, 'verdict') else str(action)
        )
        reasoning = action.reasoning if hasattr(action, 'reasoning') else ""

        self._verdict_history.append(verdict)

        # Compute reward
        reward = compute_reward(
            verdict=verdict,
            expected_verdict=self._scenario["expected_verdict"],
            deception_type=self._scenario["deception"]["type"],
            reasoning=reasoning,
            step_count=self._step_count,
        )
        self._total_reward += reward
        self._done = True  # Single-step episodes

        return self._build_observation(done=True, reward=reward)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(self, done: bool, reward=None) -> OversightObservation:
        """Convert the current scenario into an OversightObservation."""
        s = self._scenario
        claim_data = s.get("claim", {})
        patient_data = s.get("patient")

        claim = ClaimDetail(
            claim_id=claim_data.get("claim_id", 0),
            department=claim_data.get("department", ""),
            patient_id=claim_data.get("patient_id", ""),
            requested_resource=claim_data.get("requested_resource", ""),
            claimed_amount=claim_data.get("claimed_amount", 0.0),
            protocol=claim_data.get("protocol", ""),
            specialist_role=claim_data.get("specialist_role", ""),
        )

        patient = None
        if patient_data:
            patient = PatientRecord(
                patient_id=patient_data["patient_id"],
                name=patient_data["name"],
                age=patient_data["age"],
                gender=patient_data["gender"],
                blood_group=patient_data["blood_group"],
                vitals=patient_data["vitals"],
                comorbidities=s.get("visible_comorbidities", []),
                severity_index=patient_data["severity_index"],
            )

        return OversightObservation(
            claim=claim,
            patient=patient,
            department_trust=1.0,
            step_number=self._step_count,
            max_steps=1,
            prompt=s.get("prompt", ""),
            done=done,
            reward=reward,
        )

    @staticmethod
    def _parse_verdict(text: str) -> str:
        """Extract APPROVED or REJECTED from free-form text."""
        text_upper = text.upper().strip()
        if "REJECTED" in text_upper:
            return "REJECTED"
        if "APPROVED" in text_upper:
            return "APPROVED"
        return "REJECTED"
