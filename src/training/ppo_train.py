"""
GRPO Training Entry Point — Local Quick-Start

Works WITHOUT a running backend server. Uses the self-contained
PanaceaEnv and ScenarioGenerator to generate training data, then
delegates to the GRPO notebook for actual training.

Usage (local, no GPU — dataset generation + validation only):
  python -m src.training.ppo_train

Usage (Colab, with GPU — full training):
  See notebooks/panacea_grpo_training.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.env import PanaceaEnv
from src.environment.reward import compute_reward
from src.training.scenario_generator import ScenarioGenerator


def validate_environment(n_episodes: int = 20):
    """
    Run n episodes through PanaceaEnv to validate reward signals
    before sending to Colab for actual GRPO training.
    """
    env = PanaceaEnv(difficulty=3, seed=42)

    print("\n" + "=" * 60)
    print("  PANACEA — Environment Validation (Pre-Training Check)")
    print("=" * 60)

    total_reward = 0.0
    correct = 0
    type_stats: dict[str, list[bool]] = {}

    for ep in range(n_episodes):
        obs, info = env.reset()
        gt = info["ground_truth"]
        dtype = info["deception_type"]

        # Perfect agent: always matches ground truth
        if gt == "REJECTED":
            action = 2  # REJECTED
        else:
            action = 0  # APPROVED

        obs, reward, done, truncated, step_info = env.step(action)
        is_correct = step_info["is_correct"]

        total_reward += reward
        correct += int(is_correct)
        type_stats.setdefault(dtype, []).append(is_correct)

        if ep < 5:
            print(f"  Ep {ep+1:2d}: deception={dtype:10s} GT={gt:8s} "
                  f"action={step_info['action']:8s} reward={reward:+.2f} "
                  f"{'OK' if is_correct else 'WRONG'}")

    print(f"\n  Perfect agent accuracy: {correct}/{n_episodes} ({correct/n_episodes*100:.0f}%)")
    print(f"  Average reward: {total_reward/n_episodes:+.3f}")

    print(f"\n  Accuracy by deception type:")
    for dtype, results in sorted(type_stats.items()):
        acc = sum(results) / len(results) * 100
        print(f"    {dtype:12s}: {sum(results)}/{len(results)} ({acc:.0f}%)")


def generate_training_data(n: int = 500, output_path: str = None):
    """Generate training scenarios and optionally save to disk."""
    import json

    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate_dataset(n=n, difficulty=2, seed=42)

    print(f"\n  Generated {len(scenarios)} training scenarios")

    # Distribution stats
    from collections import Counter
    dist = Counter(s["deception_type"] for s in scenarios)
    for dtype, count in sorted(dist.items()):
        print(f"    {dtype}: {count} ({count/len(scenarios)*100:.0f}%)")

    if output_path:
        # Save prompts + labels for Colab upload
        data = [
            {
                "prompt": s["prompt"],
                "expected_verdict": s["ground_truth_label"],
                "deception_type": s["deception_type"],
            }
            for s in scenarios
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved to {output_path}")

    return scenarios


def test_reward_function():
    """Validate the reward function with known inputs."""
    print("\n  Reward function tests:")

    cases = [
        ("REJECTED", "REJECTED", "ghost",     "Ghost patient not found", +2.45),
        ("APPROVED", "REJECTED", "inflation", "",                        -3.05),
        ("APPROVED", "APPROVED", "none",      "",                        +0.95),
        ("REJECTED", "APPROVED", "none",      "",                        -2.05),
    ]

    for verdict, expected, dtype, reasoning, expected_reward in cases:
        reward = compute_reward(
            verdict=verdict,
            expected_verdict=expected,
            deception_type=dtype,
            reasoning=reasoning,
        )
        status = "OK" if abs(reward - expected_reward) < 0.1 else "FAIL"
        print(f"    {status}: {verdict:8s} vs {expected:8s} ({dtype:10s}) "
              f"reward={reward:+.2f} (expected ~{expected_reward:+.2f})")


if __name__ == "__main__":
    print("=" * 60)
    print("  PANACEA GRPO Training — Pre-Flight Checks")
    print("=" * 60)

    # 1. Validate reward function
    test_reward_function()

    # 2. Validate environment
    validate_environment(n_episodes=20)

    # 3. Generate training data
    generate_training_data(n=100)

    print("\n" + "=" * 60)
    print("  All checks passed! Ready for GRPO training on Colab.")
    print("  Run: notebooks/panacea_grpo_training.py")
    print("=" * 60)
