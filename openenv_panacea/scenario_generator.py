from typing import Literal

from src.training.scenario_generator import ScenarioGenerator

DeceptionType = Literal["ghost", "inflation", "masking", "collusion", "none"]


def generate_scenario(
    deception_type: DeceptionType | None = None,
    seed: int | None = None,
    gpt4o_cache_path: str | None = None,
) -> dict:
    """Generate one adversarial claim scenario for the env to serve."""
    generator = ScenarioGenerator(seed=seed, cache_path=gpt4o_cache_path)
    # Cache files already mix difficulties; otherwise force difficulty 3 so collusion can appear.
    difficulty = 3 if not gpt4o_cache_path else 2
    raw = generator.generate(difficulty=difficulty)
    raw["expected_verdict"] = raw.get("ground_truth_label", "APPROVED")
    raw["deception"] = {"type": raw.get("deception_type", "none")}
    return raw
