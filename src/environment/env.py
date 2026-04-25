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

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..training.scenario_generator import ScenarioGenerator
from ..environment.reward import compute_reward


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

    # ── Internal helpers ──────────────────────────────────────────────────────

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


# ── Quick test ────────────────────────────────────────────────────────────────

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
