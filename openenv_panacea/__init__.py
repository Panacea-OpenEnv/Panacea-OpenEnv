"""
Panacea — OpenEnv Hospital Oversight Environment

An adversarial multi-agent environment for training LLM oversight agents
to detect deception in hospital resource claims.
"""

from .models import OversightAction, OversightObservation, PanaceaEpisodeState
from .scenario_generator import generate_scenario, generate_dataset
from .reward import compute_reward, compute_reward_batch

# Client import is deferred — requires openenv-core installed
try:
    from .client import PanaceaEnv
except ImportError:
    PanaceaEnv = None  # OpenEnv not installed — client unavailable

__all__ = [
    "OversightAction",
    "OversightObservation",
    "PanaceaEpisodeState",
    "PanaceaEnv",
    "generate_scenario",
    "generate_dataset",
    "compute_reward",
    "compute_reward_batch",
]
