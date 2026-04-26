"""
Panacea OpenEnv client for connecting to the POMDP environment server.
"""

from openenv.core.env_client import EnvClient

from .models import OversightAction, OversightObservation, ToolCallAction, VerdictAction


class PanaceaEnv(EnvClient):
    """Client for the Panacea hospital POMDP oversight environment.

    Async usage:
        async with PanaceaEnv(base_url="http://localhost:8001") as client:
            result = await client.reset()
            print(result.observation.prompt)

            # Investigate first
            result = await client.call_tool("TOOL_REGISTRY")
            print(result.observation.last_tool_evidence)

            # Then commit a verdict
            result = await client.submit_verdict(
                "REJECTED", reasoning="Patient ID returned NO RECORD."
            )
            print(f"Total reward: {result.reward}")

    Sync usage:
        with PanaceaEnv(base_url="http://localhost:8001").sync() as client:
            client.reset()
            client.call_tool("TOOL_REPORTS")
            client.submit_verdict("APPROVED", reasoning="Clean.")
    """

    def __init__(self, base_url: str = "http://localhost:8001", **kwargs):
        super().__init__(
            base_url=base_url,
            action_type=OversightAction,
            observation_type=OversightObservation,
            **kwargs,
        )

    # ── Convenience helpers ──────────────────────────────────────────────────
    # These wrap the underlying client.step() with the right action shape so
    # callers don't need to remember the discriminator field.

    def call_tool(self, tool_name: str):
        """Invoke an enterprise API to gather evidence (non-terminal)."""
        return self.step(ToolCallAction(tool_name=tool_name))

    def submit_verdict(self, verdict: str, reasoning: str = ""):
        """Submit the terminal APPROVED / REJECTED decision."""
        return self.step(VerdictAction(verdict=verdict, reasoning=reasoning))
