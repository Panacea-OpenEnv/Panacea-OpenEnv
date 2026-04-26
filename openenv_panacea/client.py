from openenv.core.env_client import EnvClient

from .models import OversightAction, OversightObservation, ToolCallAction, VerdictAction


class PanaceaEnv(EnvClient):
    """Client for the Panacea POMDP env.

    Example:
        async with PanaceaEnv("http://localhost:8001") as c:
            await c.reset()
            await c.call_tool("TOOL_REGISTRY")
            await c.submit_verdict("REJECTED", "patient ID not found")
    """

    def __init__(self, base_url: str = "http://localhost:8001", **kwargs):
        super().__init__(
            base_url=base_url,
            action_type=OversightAction,
            observation_type=OversightObservation,
            **kwargs,
        )

    def call_tool(self, tool_name: str):
        return self.step(ToolCallAction(tool_name=tool_name))

    def submit_verdict(self, verdict: str, reasoning: str = ""):
        return self.step(VerdictAction(verdict=verdict, reasoning=reasoning))
