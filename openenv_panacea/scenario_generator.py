"""
Self-contained adversarial scenario generator for Panacea.

Generates realistic hospital resource-claim scenarios with embedded
deception — NO external database or API required. This makes the
environment fully portable to Google Colab for GRPO training.

This file proxies to the unified src.training.scenario_generator to ensure
collusion and other new deception types are fully supported.
"""

from typing import Literal

# Import the unified generator from src/training
from src.training.scenario_generator import ScenarioGenerator

DeceptionType = Literal["ghost", "inflation", "masking", "collusion", "none"]

def generate_scenario(
    deception_type: DeceptionType | None = None,
    seed: int | None = None,
    gpt4o_cache_path: str | None = None,
) -> dict:
    """
    Generate one adversarial hospital claim scenario.

    Args:
        deception_type: Force a specific deception type, or None for random.
        seed: Optional random seed for reproducibility.
        gpt4o_cache_path: Optional path to a JSONL file containing pre-generated
            GPT-4o reports (produced by gpt4o_harvester.py).

    Returns:
        dict: Scenario matching the OpenEnv expected schema.
    """
    generator = ScenarioGenerator(seed=seed, cache_path=gpt4o_cache_path)
    
    # We use difficulty 3 to ensure collusion is included, unless cache is provided
    # (cache inherently has its own difficulty mix, usually GPT-4o level)
    difficulty = 3 if not gpt4o_cache_path else 2
    
    # Generate the raw scenario
    raw = generator.generate(difficulty=difficulty)
    
    # Map the output back to what OpenEnv expects
    # OpenEnv historically expected "expected_verdict" and a nested "deception": {"type": ...}
    raw["expected_verdict"] = raw.get("ground_truth_label", "APPROVED")
    raw["deception"] = {"type": raw.get("deception_type", "none")}
    
    return raw
