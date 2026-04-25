import asyncio
import json
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.architecture.event_bus import event_bus, ActionProposedEvent, VerificationDecisionEvent, ToolExecutedEvent
from src.database.mongo_client import get_patient_comorbidities, get_patient_vitals

class OversightAgent:
    """
    Tool-enabled Evaluator. Listens for proposed claims, calls MongoDB tools
    to check reality, and publishes verdicts.
    """
    def __init__(self, agent_id: str = "OversightCore"):
        self.agent_id = agent_id
        
    async def start_listening(self):
        print(f"[{self.agent_id}] Online. Subscribing to agent.action.proposed...")
        queue = event_bus.subscribe("agent.action.proposed")
        
        while True:
            event: ActionProposedEvent = await queue.get()
            asyncio.create_task(self._process_claim(event))
            
    async def _process_claim(self, claim_event: ActionProposedEvent):
        print(f"[{self.agent_id}] Received claim from {claim_event.agent_id}. Initializing review...")
        
        patient_id = claim_event.claimed_state.get('patient_id')
        
        # 1. Data Gathering from MongoDB
        try:
            comorbidities = await get_patient_comorbidities(patient_id)
            vitals = await get_patient_vitals(patient_id)
            db_data = {
                "comorbidities": comorbidities,
                "vitals": vitals
            }
            db_result_str = json.dumps(db_data, indent=2)
            
            await event_bus.publish(ToolExecutedEvent(
                agent_id=self.agent_id, tool_name="fetch_mongodb_records", arguments=patient_id, result="Success"
            ))
            
        except Exception as e:
            db_result_str = f"Error fetching from MongoDB: {e}"
            
        # 2. Decision Logic
        # (This is currently deterministic mock logic, but operates on MongoDB JSON)
        verdict = "APPROVED"
        reasoning = "Verified cleanly."
        
        if "Error fetching" in db_result_str:
            verdict = "REJECTED"
            reasoning = "Suspicious backend state tracking timeout."
        elif "Hemophilia" in db_result_str:
            verdict = "REJECTED"
            reasoning = "Caught omission of Hemophilia record."
            
        print(f"[{self.agent_id}] Finished review. Verdict: {verdict}")
        
        # 3. Publish Decision
        decision_event = VerificationDecisionEvent(
            agent_id=self.agent_id,
            target_action_id=claim_event.event_id,
            verdict=verdict,
            reasoning=reasoning
        )
        await event_bus.publish(decision_event)
