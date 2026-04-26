"""
PanaceaEnv — Self-Contained Gym Environment for RL Training

Does NOT require localhost:8000 or any backend server.
Uses ScenarioGenerator directly to produce episodes, and scores
the oversight agent's actions using the arena.py reward table.

Supports OpenEnv observation format: prompt string + metadata.

Usage:
    env = PanaceaEnv(difficulty=2)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action=2)  # REJECTED
"""

import json
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..training.scenario_generator import ScenarioGenerator
from ..environment.reward import compute_reward
from .tool_backends import TOOL_BACKENDS, TOOL_NAMES, call_tool, tool_cost

# Where the curriculum log lands. Override via env var for unit tests.
CURRICULUM_LOG_PATH = os.environ.get(
    "PANACEA_CURRICULUM_LOG", "data/curriculum_log.jsonl"
)


class PanaceaEnv(gym.Env):
    """
    Gym-compatible environment for training oversight agents.

    observation_space: Dict with prompt (text) + trust score + step count
    action_space:      Discrete(3) — 0=APPROVED, 1=PARTIAL, 2=REJECTED
    """

    metadata = {"render_modes": ["human"]}
    MAX_EPISODE_STEPS = 6

    ACTION_MAP = {0: "APPROVED", 1: "PARTIAL", 2: "REJECTED"}

    def __init__(self, difficulty: int = 1, seed: int | None = None):
        super().__init__()
        self.difficulty = difficulty
        self.generator = ScenarioGenerator(seed=seed)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "prompt": spaces.Text(max_length=4096),
            "department_trust": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        # Dynamic trust ledger (reused from original env.py)
        self.trust_ledger: dict[str, float] = {}
        for spec_name in ["Cardiology", "Neurology", "Pulmonology", "Oncology",
                          "Orthopedics", "General Medicine", "Nephrology",
                          "Infectious Disease", "Gastroenterology", "Hematology",
                          "Neurosurgery", "Pediatrics", "Gynecology", "Obstetrics",
                          "Dermatology", "Ophthalmology", "Otolaryngology",
                          "Urology", "Endocrinology", "Rheumatology",
                          "Psychiatry", "Radiology", "Anesthesiology",
                          "Pathology", "Plastic Surgery", "Vascular Surgery"]:
            self.trust_ledger[spec_name] = 1.0

        self._scenario: dict = {}
        self.step_count: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Decay trust towards 1.0 between episodes
        self._decay_trust()

        self._scenario = self.generator.generate(difficulty=self.difficulty)
        obs = self._get_obs()
        info = {
            "deception_type": self._scenario["deception_type"],
            "ground_truth": self._scenario["ground_truth_label"],
            "difficulty": self._scenario["difficulty"],
            "department": self._scenario["department"],
            "num_reports": len(self._scenario.get("reports", [])),
        }
        return obs, info

    def step(self, action: int):
        """
        Score the oversight agent's action against ground truth.

        Args:
            action: 0=APPROVED, 1=PARTIAL, 2=REJECTED

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Context exhaustion: force auto-reject with heavy penalty
        if self.step_count >= self.MAX_EPISODE_STEPS:
            reward = -2.0  # Timeout penalty
            obs = self._get_obs()
            return obs, reward, True, True, {"status": "timeout_exhaustion"}

        action_str = self.ACTION_MAP.get(action, "REJECTED")

        reward = compute_reward(
            verdict=action_str,
            expected_verdict=self._scenario["ground_truth_label"],
            deception_type=self._scenario["deception_type"],
            step_count=self.step_count,
        )

        # Trust penalty: update ledger based on decision correctness
        dept = self._scenario.get("department", "Unknown")
        is_correct = (action_str == self._scenario["ground_truth_label"])

        if not is_correct:
            self._penalize_trust(dept)

        info = {
            "action": action_str,
            "is_correct": is_correct,
            "deception_type": self._scenario["deception_type"],
            "ground_truth": self._scenario["ground_truth_label"],
            "department": dept,
            "department_trust": self.trust_ledger.get(dept, 1.0),
            "step_count": self.step_count,
            "claimed_amount": self._scenario.get("claimed_amount", 0),
            "expected_cost": self._scenario.get("expected_cost", 0),
            "fraud_details": self._scenario.get("fraud_details", {}),
        }

        obs = self._get_obs()
        return obs, reward, True, False, info

    def render(self):
        dept = self._scenario.get("department", "Unknown")
        trust = self.trust_ledger.get(dept, 1.0)
        dtype = self._scenario.get("deception_type", "unknown")
        label = self._scenario.get("ground_truth_label", "?")
        print(
            f"-- Step {self.step_count}/{self.MAX_EPISODE_STEPS} | "
            f"Dept: {dept} (Trust: {trust:.2f}) | "
            f"Deception: {dtype} | Label: {label} --"
        )

    #  Internal helpers 

    def _get_obs(self) -> dict:
        dept = self._scenario.get("department", "Unknown")
        trust = self.trust_ledger.get(dept, 1.0)
        return {
            "prompt": self._scenario.get("prompt", ""),
            "department_trust": np.array([trust], dtype=np.float32),
        }

    def _decay_trust(self):
        """Exponential trust recovery toward 1.0 (Boy Who Cried Wolf)."""
        decay_rate = 0.1
        for dept in self.trust_ledger:
            self.trust_ledger[dept] += (1.0 - self.trust_ledger[dept]) * decay_rate

    def _penalize_trust(self, department: str, amount: float = 0.3):
        if department in self.trust_ledger:
            self.trust_ledger[department] = max(0.0, self.trust_ledger[department] - amount)


#  POMDP environment (multi-step investigation) 


class PanaceaPOMDPEnv(gym.Env):
    """
    Partially-observable, multi-step oversight environment.

    At t=0 the agent sees only the patient ID, claim amount, and department.
    To accumulate evidence it must invoke enterprise APIs (TOOL_REGISTRY,
    TOOL_VITALS, TOOL_REPORTS, TOOL_DRUGS, TOOL_BILLING) — each with a cost
    and an independent reliability. Episode ends when the agent emits
    APPROVE or REJECT (or hits the step budget).

    Action space (Discrete(7)):
        0  APPROVE   (terminal)
        1  REJECT    (terminal)
        2  TOOL_REGISTRY
        3  TOOL_VITALS
        4  TOOL_REPORTS
        5  TOOL_DRUGS
        6  TOOL_BILLING

    Observation:
        prompt:        accumulated context string (grows as tools are called)
        tools_used:    Box(5,) — 0/1 mask of which tools were invoked
        step_count:    Discrete scalar
        department_trust: Box(1,) — current trust score for the department
    """

    metadata = {"render_modes": ["human"]}
    MAX_STEPS = 8

    APPROVE = 0
    REJECT = 1
    TOOL_ACTION_OFFSET = 2  # actions 2..2+len(TOOL_NAMES)-1 are tool calls

    def __init__(self, difficulty: int = 1, seed: int | None = None,
                 max_steps: int | None = None, adaptive: bool = False,
                 adaptive_window: int = 50,
                 curriculum_log_path: str | None = None,
                 curriculum_log_every: int = 50):
        super().__init__()
        self.difficulty = difficulty
        self.generator = ScenarioGenerator(seed=seed)
        if max_steps is not None:
            self.MAX_STEPS = max_steps

        self.adaptive_sampler = None
        if adaptive:
            from ..training.adaptive_adversary import AdaptiveDeceptionSampler
            self.adaptive_sampler = AdaptiveDeceptionSampler(
                window=adaptive_window, seed=seed,
            )

        # Curriculum drift log — episode-level snapshots of the adaptive sampler.
        # Off by default for non-adaptive runs; auto-on whenever adaptive=True.
        self.curriculum_log_path = (
            curriculum_log_path if curriculum_log_path is not None
            else (CURRICULUM_LOG_PATH if adaptive else None)
        )
        self.curriculum_log_every = curriculum_log_every
        self._episode_counter = 0

        n_tools = len(TOOL_NAMES)
        self.action_space = spaces.Discrete(2 + n_tools)
        self.observation_space = spaces.Dict({
            "prompt": spaces.Text(max_length=8192),
            "tools_used": spaces.Box(low=0, high=1, shape=(n_tools,), dtype=np.int8),
            "step_count": spaces.Box(low=0, high=self.MAX_STEPS, shape=(1,), dtype=np.int32),
            "department_trust": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        # Trust ledger reused from the single-step env's spec list
        self.trust_ledger: dict[str, float] = {dept: 1.0 for dept in [
            "Cardiology", "Neurology", "Pulmonology", "Oncology",
            "Orthopedics", "General Medicine", "Nephrology",
            "Infectious Disease", "Gastroenterology", "Hematology",
            "Neurosurgery", "Pediatrics", "Gynecology", "Obstetrics",
            "Dermatology", "Ophthalmology", "Otolaryngology",
            "Urology", "Endocrinology", "Rheumatology",
            "Psychiatry", "Radiology", "Anesthesiology",
            "Pathology", "Plastic Surgery", "Vascular Surgery",
        ]}

        self._scenario: dict = {}
        self._context: str = ""
        self._tools_used: np.ndarray = np.zeros(n_tools, dtype=np.int8)
        self.step_count: int = 0
        self._tool_cost_total: float = 0.0
        self._rng = np.random.default_rng(seed)

    #  Gym API 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._tool_cost_total = 0.0
        self._tools_used = np.zeros(len(TOOL_NAMES), dtype=np.int8)
        self._decay_trust()

        forced_type = self.adaptive_sampler.sample() if self.adaptive_sampler else None
        self._scenario = self.generator.generate(
            difficulty=self.difficulty, deception_type=forced_type,
        )
        self._context = self._initial_context(self._scenario)

        info = {
            "deception_type": self._scenario["deception_type"],
            "ground_truth": self._scenario["ground_truth_label"],
            "difficulty": self._scenario.get("difficulty", self.difficulty),
            "department": self._scenario["department"],
            "tool_names": list(TOOL_NAMES),
        }
        return self._get_obs(), info

    def step(self, action: int):
        self.step_count += 1
        dept = self._scenario.get("department", "Unknown")
        gt = self._scenario["ground_truth_label"]

        # Step-budget exhaustion
        if self.step_count > self.MAX_STEPS:
            return self._get_obs(), -2.0, True, True, {
                "status": "timeout_exhaustion",
                "tool_cost_total": self._tool_cost_total,
                "ground_truth": gt,
            }

        # Tool action — accumulate evidence, episode continues
        if action >= self.TOOL_ACTION_OFFSET:
            tool_idx = action - self.TOOL_ACTION_OFFSET
            if tool_idx >= len(TOOL_NAMES):
                # Invalid action — penalize and end
                return self._get_obs(), -1.0, True, False, {
                    "status": "invalid_action",
                    "ground_truth": gt,
                }
            tool_name = TOOL_NAMES[tool_idx]

            # Repeat-call penalty: discourage spamming the same API
            already = bool(self._tools_used[tool_idx])
            self._tools_used[tool_idx] = 1

            evidence = call_tool(tool_name, self._scenario, rng=None)
            self._context += f"\n\n>> CALL {tool_name} ({TOOL_BACKENDS[tool_name]['app']})\n{evidence}"

            cost = tool_cost(tool_name)
            if already:
                cost += -0.05  # small repeat penalty
            self._tool_cost_total += cost

            return self._get_obs(), cost, False, False, {
                "status": "tool_call",
                "tool": tool_name,
                "tool_repeat": already,
                "tool_cost": cost,
                "tool_cost_total": self._tool_cost_total,
                "ground_truth": gt,
            }

        # Terminal verdict — pass tool outputs as evidence so the +0.5 bonus
        # is awarded only when the agent actually called the right tool AND
        # the tool returned the canonical evidence flag (uncheatable).
        verdict = "APPROVED" if action == self.APPROVE else "REJECTED"
        accuracy_reward = compute_reward(
            verdict=verdict,
            expected_verdict=gt,
            deception_type=self._scenario["deception_type"],
            reasoning=self._context,
            step_count=self.step_count,
            evidence={"tool_outputs": self._context},
        )
        reward = accuracy_reward + self._tool_cost_total

        is_correct = (verdict == gt)
        if not is_correct:
            self._penalize_trust(dept)

        if self.adaptive_sampler is not None:
            dtype = self._scenario["deception_type"]
            detected = (verdict == "REJECTED") if dtype != "none" else (verdict == "APPROVED")
            self.adaptive_sampler.record(dtype, detected)

        # Episode-level curriculum-drift snapshot (every N completed episodes)
        self._episode_counter += 1
        if (
            self.adaptive_sampler is not None
            and self.curriculum_log_path
            and self._episode_counter % self.curriculum_log_every == 0
        ):
            self._write_curriculum_snapshot()

        info = {
            "status": "verdict",
            "action": verdict,
            "is_correct": is_correct,
            "deception_type": self._scenario["deception_type"],
            "ground_truth": gt,
            "department": dept,
            "department_trust": self.trust_ledger.get(dept, 1.0),
            "step_count": self.step_count,
            "accuracy_reward": accuracy_reward,
            "tool_cost_total": self._tool_cost_total,
            "tools_used": self._tools_used.copy(),
        }
        return self._get_obs(), reward, True, False, info

    def render(self):
        print(f"-- step {self.step_count}/{self.MAX_STEPS} | "
              f"tools_used={self._tools_used.tolist()} | "
              f"tool_cost={self._tool_cost_total:+.2f} --")
        print(self._context[-600:])

    #  Internal helpers 

    def _initial_context(self, episode: dict) -> str:
        """Minimal info revealed at t=0 — the agent must probe for the rest."""
        tool_menu = "\n".join(
            f"  - action={i + self.TOOL_ACTION_OFFSET} {name} "
            f"({TOOL_BACKENDS[name]['app']}, cost≈{tool_cost(name):+.2f}, "
            f"reliability={TOOL_BACKENDS[name]['reliability']})"
            for i, name in enumerate(TOOL_NAMES)
        )
        return (
            "You are a hospital oversight agent investigating a resource claim.\n"
            "You see only the headline below. To gather evidence, choose a tool "
            "action — each enterprise API has its own reliability and cost. "
            "When you have enough evidence, output APPROVE (action=0) or "
            "REJECT (action=1).\n\n"
            f"CLAIM HEADLINE:\n"
            f"  patient_id     = {episode.get('patient_id', 'UNKNOWN')}\n"
            f"  department     = {episode.get('department', 'Unknown')}\n"
            f"  claimed_amount = ${episode.get('claimed_amount', 0):,.2f}\n\n"
            f"AVAILABLE TOOLS:\n{tool_menu}\n"
            f"\nEVIDENCE ACCUMULATED:\n  (none yet — call a tool)"
        )

    def _get_obs(self) -> dict:
        dept = self._scenario.get("department", "Unknown")
        return {
            "prompt": self._context,
            "tools_used": self._tools_used.copy(),
            "step_count": np.array([self.step_count], dtype=np.int32),
            "department_trust": np.array(
                [self.trust_ledger.get(dept, 1.0)], dtype=np.float32
            ),
        }

    def _decay_trust(self):
        for d in self.trust_ledger:
            self.trust_ledger[d] += (1.0 - self.trust_ledger[d]) * 0.1

    def _penalize_trust(self, department: str, amount: float = 0.3):
        if department in self.trust_ledger:
            self.trust_ledger[department] = max(
                0.0, self.trust_ledger[department] - amount
            )

    def _write_curriculum_snapshot(self) -> None:
        """Append one snapshot row to the curriculum drift log."""
        if self.adaptive_sampler is None or not self.curriculum_log_path:
            return
        snap = self.adaptive_sampler.snapshot()
        record = {
            "episode": self._episode_counter,
            "weights": snap["weights"],
            "detection_rates": snap["detection_rates"],
            "samples": snap["samples"],
        }
        try:
            os.makedirs(os.path.dirname(self.curriculum_log_path) or ".", exist_ok=True)
            with open(self.curriculum_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            # Logging failures must never abort training
            pass


# Backward-compat alias for callers that want the original single-step env
PanaceaEnvSingleStep = PanaceaEnv


#  Quick test 

if __name__ == "__main__":
    env = PanaceaEnv(difficulty=2, seed=42)

    for ep in range(5):
        obs, info = env.reset()
        print(f"\nEpisode {ep+1}: {info['deception_type']} | GT={info['ground_truth']}")

        # Simulate a "perfect" agent
        if info["ground_truth"] == "REJECTED":
            action = 2  # REJECTED
        else:
            action = 0  # APPROVED

        obs, reward, done, truncated, step_info = env.step(action)
        print(f"  Action={step_info['action']} | Correct={step_info['is_correct']} | Reward={reward:+.2f}")
