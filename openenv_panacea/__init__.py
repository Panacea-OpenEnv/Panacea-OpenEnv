"""
Panacea — OpenEnv Hospital Oversight Environment

An adversarial multi-agent environment for training LLM oversight agents
to detect deception in hospital resource claims.
"""

from .models import (
    OversightAction,
    OversightObservation,
    PanaceaEpisodeState,
    ToolCallAction,
    VerdictAction,
)
from .scenario_generator import generate_scenario
from .reward import compute_reward, compute_reward_batch
from .tool_backends import TOOL_BACKENDS, TOOL_NAMES, call_tool, tool_cost

# Client import is deferred — requires openenv-core installed
try:
    from .client import PanaceaEnv
except ImportError:
    PanaceaEnv = None

__all__ = [
    "OversightAction",
    "OversightObservation",
    "PanaceaEpisodeState",
    "ToolCallAction",
    "VerdictAction",
    "PanaceaEnv",
    "generate_scenario",
    "compute_reward",
    "compute_reward_batch",
    "TOOL_BACKENDS",
    "TOOL_NAMES",
    "call_tool",
    "tool_cost",
]
