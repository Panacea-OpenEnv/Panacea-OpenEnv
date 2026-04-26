"""
Panacea OpenEnv POMDP Environment (Server-Side).

Multi-step partially-observable oversight env. Each episode:
  1. reset()  → returns minimal headline (patient_id, claim_amount, department)
  2. step(action)
       - tool_call: invoke an enterprise API; appends evidence to `prompt`,
                    returns small per-call cost as reward, done=False
       - verdict:   final APPROVE/REJECT; computes accuracy + evidence bonus,
                    returns total reward, done=True
  3. state()  → episode metadata for telemetry

Mirrors src/environment/env.py PanaceaPOMDPEnv exactly.
"""

import uuid

from openenv.core import Environment

from ..models import (
    OversightAction,
    OversightObservation,
    ClaimDetail,
    PatientRecord,
)
from ..scenario_generator import generate_scenario
from ..reward import compute_reward
from ..tool_backends import (
    TOOL_BACKENDS,
    TOOL_NAMES,
    call_tool,
    tool_cost,
)


MAX_STEPS = 8
REPEAT_TOOL_PENALTY = -0.05


class PanaceaEnvironment(Environment):
    """Adversarial hospital POMDP oversight environment.

    Action menu:
      type='tool_call' + tool_name=TOOL_REGISTRY|TOOL_VITALS|TOOL_REPORTS|
                                    TOOL_DRUGS|TOOL_BILLING
      type='verdict'   + verdict=APPROVED|REJECTED + reasoning=<text>
    """

    def __init__(self):
        super().__init__()
        self._episode_id: str = ""
        self._step_count: int = 0
        self._scenario: dict = {}
        self._context: str = ""
        self._tools_used: dict[str, bool] = {}
        self._tool_cost_total: float = 0.0
        self._verdict_history: list[str] = []
        self._done: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> OversightObservation:
        self._episode_id = episode_id or str(uuid.uuid4())[:12]
        self._step_count = 0
        self._tool_cost_total = 0.0
        self._tools_used = {name: False for name in TOOL_NAMES}
        self._verdict_history = []
        self._done = False

        self._scenario = generate_scenario(seed=seed)
        self._context = self._initial_context(self._scenario)

        return self._build_observation(done=False, reward=None, last_evidence="")

    def step(self, action: OversightAction, timeout_s=None, **kwargs) -> OversightObservation:
        if self._done:
            return self._build_observation(done=True, reward=None, last_evidence="")

        self._step_count += 1

        # Step-budget exhaustion
        if self._step_count > MAX_STEPS:
            self._done = True
            return self._build_observation(
                done=True, reward=-2.0, last_evidence="",
            )

        # Determine action type — back-compat: legacy callers pass only verdict
        action_type = getattr(action, "type", None) or "verdict"

        # ── Tool call: gather evidence, episode continues ────────────────────
        if action_type == "tool_call":
            tool_name = (action.tool_name or "").upper()
            if tool_name not in TOOL_BACKENDS:
                self._done = True
                return self._build_observation(
                    done=True, reward=-1.0,
                    last_evidence=f"INVALID TOOL: {tool_name!r}",
                )

            already = self._tools_used.get(tool_name, False)
            self._tools_used[tool_name] = True

            evidence = call_tool(tool_name, self._scenario, rng=None)
            self._context += (
                f"\n\n>> CALL {tool_name} ({TOOL_BACKENDS[tool_name]['app']})\n{evidence}"
            )

            cost = tool_cost(tool_name)
            if already:
                cost += REPEAT_TOOL_PENALTY
            self._tool_cost_total += cost

            return self._build_observation(
                done=False, reward=cost, last_evidence=evidence,
            )

        # ── Verdict: terminal scoring ────────────────────────────────────────
        verdict = self._parse_verdict(action.verdict)
        self._verdict_history.append(verdict)

        gt = self._scenario.get("expected_verdict", "APPROVED")
        deception_type = self._scenario.get("deception", {}).get("type", "none")

        accuracy_reward = compute_reward(
            verdict=verdict,
            expected_verdict=gt,
            deception_type=deception_type,
            reasoning=action.reasoning,
            step_count=self._step_count,
            evidence={"tool_outputs": self._context},
        )
        total_reward = round(accuracy_reward + self._tool_cost_total, 4)
        self._done = True

        obs = self._build_observation(
            done=True, reward=total_reward, last_evidence="",
        )
        # Surface ground truth + reward breakdown via metadata for telemetry
        obs.metadata.update({
            "ground_truth":       gt,
            "deception_type":     deception_type,
            "accuracy_reward":    accuracy_reward,
            "tool_cost_total":    round(self._tool_cost_total, 4),
            "verdict":            verdict,
            "is_correct":         verdict == gt,
            "tools_used":         {k: bool(v) for k, v in self._tools_used.items()},
        })
        return obs

    def state(self) -> dict:
        return {
            "episode_id":       self._episode_id,
            "step_count":       self._step_count,
            "done":             self._done,
            "tools_used":       dict(self._tools_used),
            "tool_cost_total":  round(self._tool_cost_total, 4),
            "verdict_history":  list(self._verdict_history),
            "deception_type":   self._scenario.get("deception", {}).get("type", "none"),
            "expected_verdict": self._scenario.get("expected_verdict", "APPROVED"),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _initial_context(self, episode: dict) -> str:
        tool_menu = "\n".join(
            f"  - {name} ({TOOL_BACKENDS[name]['app']}, cost≈{tool_cost(name):+.2f}, "
            f"reliability={TOOL_BACKENDS[name]['reliability']})"
            for name in TOOL_NAMES
        )
        claim = episode.get("claim", {}) or {}
        return (
            "You are a hospital oversight agent investigating a resource claim.\n"
            "You see only the headline below. To gather evidence, send a tool_call "
            "action — each enterprise API has its own reliability and cost. When "
            "you have enough evidence, send a verdict action (APPROVED / REJECTED).\n\n"
            "CLAIM HEADLINE:\n"
            f"  patient_id     = {claim.get('patient_id', episode.get('patient_id', 'UNKNOWN'))}\n"
            f"  department     = {claim.get('department', episode.get('department', 'Unknown'))}\n"
            f"  claimed_amount = ${claim.get('claimed_amount', episode.get('claimed_amount', 0)):,.2f}\n\n"
            f"AVAILABLE TOOLS:\n{tool_menu}\n"
            f"\nEVIDENCE ACCUMULATED:\n  (none yet — call a tool)"
        )

    def _build_observation(
        self, done: bool, reward, last_evidence: str,
    ) -> OversightObservation:
        s = self._scenario
        claim_data = s.get("claim", {}) or {}
        patient_data = s.get("patient")

        claim = ClaimDetail(
            claim_id=claim_data.get("claim_id", 0),
            department=claim_data.get("department", s.get("department", "")),
            patient_id=claim_data.get("patient_id", s.get("patient_id", "")),
            requested_resource=claim_data.get("requested_resource", ""),
            claimed_amount=claim_data.get("claimed_amount", s.get("claimed_amount", 0.0)),
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
            max_steps=MAX_STEPS,
            prompt=self._context,
            tools_used=dict(self._tools_used),
            tool_cost_total=round(self._tool_cost_total, 4),
            last_tool_evidence=last_evidence,
            done=done,
            reward=reward,
        )

    @staticmethod
    def _parse_verdict(text: str) -> str:
        text_upper = (text or "").upper().strip()
        if "REJECTED" in text_upper:
            return "REJECTED"
        if "APPROVED" in text_upper:
            return "APPROVED"
        return "REJECTED"
