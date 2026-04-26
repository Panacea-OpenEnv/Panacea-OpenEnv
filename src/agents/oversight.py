"""
Oversight Agent — Panacea Hospital

Tool-enabled Evaluator that listens for proposed claims on the event bus,
verifies them against MongoDB ground truth, and publishes verdicts.

When OVERSIGHT_ENDPOINT is set in .env, calls the trained RL model for
intelligent LLM reasoning. Otherwise, falls back to deterministic
MongoDB-checking rules (original behavior preserved).
"""

import asyncio
import json
import os
from typing import Dict, Any

from ..architecture.event_bus import (
    event_bus,
    ActionProposedEvent,
    VerificationDecisionEvent,
    ToolExecutedEvent,
)
from ..database.mongo_client import get_patient_comorbidities, get_patient_vitals
from ..inference.inference_server import query_oversight_model


class OversightAgent:
    """
    Tool-enabled Evaluator. Listens for proposed claims, calls MongoDB tools
    to check reality, and publishes verdicts.

    Mode selection:
      - If OVERSIGHT_ENDPOINT is set: uses the trained RL model
      - Otherwise: uses deterministic MongoDB-checking logic (fallback)
    """

    def __init__(self, agent_id: str = "OversightCore"):
        self.agent_id = agent_id
        self._use_rl_model = bool(os.getenv("OVERSIGHT_ENDPOINT"))

    async def start_listening(self):
        print(f"[{self.agent_id}] Online (mode={'RL Model' if self._use_rl_model else 'Deterministic'}). "
              f"Subscribing to agent.action.proposed...")
        queue = event_bus.subscribe("agent.action.proposed")

        while True:
            event: ActionProposedEvent = await queue.get()
            asyncio.create_task(self._process_claim(event))

    async def _process_claim(self, claim_event: ActionProposedEvent):
        print(f"[{self.agent_id}] Received claim from {claim_event.agent_id}. Initializing review...")

        patient_id = claim_event.claimed_state.get("patient_id")

        # Data Gathering from MongoDB (always — provides context for both paths)
        db_data = await self._fetch_db_data(patient_id)

        # Decision Logic
        from .oversight_core import verify_claim
        claimed = claim_event.claimed_state
        
        verify_result = verify_claim(
            patient_id=patient_id,
            reports=[], # No parsed reports in event bus yet
            resource_requests=[{"resources": [claimed.get("requested_resource")], "specialist": claimed.get("department")}] if claimed.get("requested_resource") else [],
            patient_from_db=db_data.get("vitals") or db_data.get("comorbidities"),
            db_comorbidities=db_data.get("comorbidities", []),
            claimed_comorbidities=claimed.get("comorbidities", []),
            true_state=str(claim_event.true_patient_state),
            claimed_state_str=str(claimed),
        )

        verdict = verify_result["decision"]
        reasoning = verify_result["reasoning"]

        print(f"[{self.agent_id}] Finished review. Verdict: {verdict}")

        # Publish Decision
        decision_event = VerificationDecisionEvent(
            agent_id=self.agent_id,
            target_action_id=claim_event.event_id,
            verdict=verdict,
            reasoning=reasoning,
        )
        await event_bus.publish(decision_event)

    #  Data fetch 

    async def _fetch_db_data(self, patient_id: str) -> dict:
        """Fetch patient data from MongoDB for verification."""
        try:
            comorbidities = await get_patient_comorbidities(patient_id)
            vitals = await get_patient_vitals(patient_id)
            db_data = {
                "comorbidities": comorbidities,
                "vitals": vitals,
            }

            await event_bus.publish(ToolExecutedEvent(
                agent_id=self.agent_id,
                tool_name="fetch_mongodb_records",
                arguments=patient_id,
                result="Success",
            ))
            return db_data

        except Exception as e:
            return {"error": f"Error fetching from MongoDB: {e}"}


