"""Panacea — adversarial hospital oversight environment for OpenEnv."""

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
