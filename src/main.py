import asyncio

from src.backend.fhir_database import main as init_db
from src.architecture.event_bus import event_bus
from src.agents.department import AdversarialDepartmentAgent
from src.agents.oversight import OversightAgent
from src.training.arena import SelfPlayArena

async def launch_system():
    print("Initializing Panacea Adaptive Self-Play Architecture...")
    
    # Start the Referee
    arena = SelfPlayArena()
    asyncio.create_task(arena.start_listening())
    
    # Start the Defense (Oversight)
    oversight = OversightAgent()
    asyncio.create_task(oversight.start_listening())
    
    # Start the Offense (Adversary on Level 2 Curriculum = Has Schema Drift powers)
    adversary_1 = AdversarialDepartmentAgent("CardiologyAgent", action_space_level=2)
    asyncio.create_task(adversary_1.run_operations())
    
    # Keep main thread alive
    print("\n--- ALL AGENTS DEPLOYED. EVENT BUS ACTIVE ---\n")
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    # Ensure DB is fresh before starting the asynchronous loop
    asyncio.run(init_db())
    asyncio.run(launch_system())
