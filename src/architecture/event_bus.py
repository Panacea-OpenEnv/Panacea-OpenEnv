import asyncio
import uuid
from typing import Dict, Any, List, Callable
from pydantic import BaseModel, Field

# --- Event Schemas ---

class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=asyncio.get_event_loop().time)
    topic: str
    
class ActionProposedEvent(BaseEvent):
    """Fired by the Sub-Agent proposing a claim."""
    topic: str = "agent.action.proposed"
    agent_id: str
    true_patient_state: Dict[str, Any]  # Ground truth state (invisible to oversight)
    claimed_state: Dict[str, Any]       # The fabricated/manipulated state
    tools_used: List[str]               # Tools the sub-agent used to attack (e.g. drift)

class ToolExecutedEvent(BaseEvent):
    """Fired when an agent executes a DB tool."""
    topic: str = "agent.tool.executed"
    agent_id: str
    tool_name: str
    arguments: str
    result: str

class VerificationDecisionEvent(BaseEvent):
    """Fired by Oversight Agent when rendering a verdict."""
    topic: str = "agent.verification.complete"
    agent_id: str
    target_action_id: str               # Links back to ActionProposedEvent
    verdict: str                        # "APPROVED" or "REJECTED"
    reasoning: str

class MatchResultEvent(BaseEvent):
    """Fired by the Referee/Arena after evaluating the zero-sum game."""
    topic: str = "arena.match.result"
    oversight_reward: float
    sub_agent_reward: float
    winner: str

# --- Pub/Sub Event Bus ---

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        
    def subscribe(self, topic: str) -> asyncio.Queue:
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        queue = asyncio.Queue()
        self.subscribers[topic].append(queue)
        return queue
        
    async def publish(self, event: BaseEvent):
        if event.topic in self.subscribers:
            for queue in self.subscribers[event.topic]:
                await queue.put(event)

# Global singleton router
event_bus = EventBus()
