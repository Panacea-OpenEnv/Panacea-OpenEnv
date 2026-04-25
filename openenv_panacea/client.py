"""
Panacea OpenEnv client for connecting to the environment server.
"""

from openenv.core.env_client import EnvClient
from .models import OversightAction, OversightObservation


class PanaceaEnv(EnvClient):
    """
    Client for the Panacea hospital oversight environment.

    Usage (async):
        async with PanaceaEnv(base_url="http://localhost:8001") as client:
            result = await client.reset()
            print(result.observation.prompt)
            result = await client.step(OversightAction(
                verdict="REJECTED",
                reasoning="Patient not found in registry"
            ))
            print(f"Reward: {result.reward}")

    Usage (sync):
        with PanaceaEnv(base_url="http://localhost:8001").sync() as client:
            result = client.reset()
            result = client.step(OversightAction(verdict="APPROVED", reasoning="Clean claim"))
    """

    def __init__(self, base_url: str = "http://localhost:8001", **kwargs):
        super().__init__(
            base_url=base_url,
            action_type=OversightAction,
            observation_type=OversightObservation,
            **kwargs,
        )
