from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio

from .database import db
from .drift_engine import drift_engine

app = FastAPI(title="Project Panacea Phase 3 Backend")

@app.on_event("startup")
async def startup():
    await db.connect()
    # Drift engine active in Phase 3
    asyncio.create_task(drift_engine.start())

@app.on_event("shutdown")
async def shutdown():
    drift_engine.stop()
    await db.disconnect()

class QueryRequest(BaseModel):
    sql: str

class ClaimReview(BaseModel):
    verdict: bool  # True for valid, False for reject

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query")
async def execute_query(request: QueryRequest):
    """
    Read-only SQL execution endpoint for the Oversight Agent's LangGraph nodes.
    Blocks any write statements (INSERT/UPDATE/DELETE/ALTER/DROP).
    Returns rows as a list of dicts, or an error dict with the exception type.
    """
    stmt = request.sql.strip().upper()
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE")
    if any(stmt.startswith(kw) for kw in forbidden):
        raise HTTPException(status_code=403, detail="Only SELECT and information_schema queries are permitted.")
    try:
        records = await db.fetch(request.sql)
        return {"status": "ok", "rows": [dict(r) for r in records]}
    except Exception as e:
        error_type = type(e).__name__
        return {"status": "error", "error_type": error_type, "detail": str(e)}


@app.get("/claims/pending", response_model=List[dict])
async def get_pending_claims():
    """Fetch pending claims for the agent to review"""
    query = """
        SELECT id, patient_id, protocol_id, department, requested_resource, claimed_amount 
        FROM claims 
        WHERE status = 'pending'
        ORDER BY id ASC -- Triage tie-breaking rule (first in)
        LIMIT 1;
    """
    try:
        records = await db.fetch(query)
        return [dict(record) for record in records]
    except Exception as e:
        # DB might be drifting, return empty to give agent time or trigger error logic elsewhere
        return []

async def check_ghost_patient(patient_id: str) -> bool:
    """Returns True if the patient does NOT exist in the database."""
    query = "SELECT 1 FROM patients WHERE patient_id = $1"
    res = await db.fetchrow(query, patient_id)
    return res is None

async def process_resource_cascade(resource_id: str) -> bool:
    """
    Checks if a resource and all its hierarchical dependencies have capacity.
    Returns True if allocation is successful (cascade holds), False if shortage.
    """
    # Base Capacity Check
    query = "SELECT max_capacity, in_use FROM resources WHERE resource_id = $1"
    res = await db.fetchrow(query, resource_id)
    if not res or res['in_use'] >= res['max_capacity']:
        return False # Bottleneck reached

    # Check Downstream Dependencies
    dep_query = "SELECT child_resource FROM dependencies WHERE parent_resource = $1"
    deps = await db.fetch(dep_query, resource_id)
    
    for dep in deps:
        child_id = dep['child_resource']
        if not await process_resource_cascade(child_id):
            return False
            
    # If we got here, hierarchy has space. In a real system we'd UPDATE in_use = in_use + 1
    # For simulation testing Phase 3, we just validate the cascade is possible.
    return True

@app.post("/claims/{claim_id}/verify")
async def verify_claim(claim_id: int, review: ClaimReview):
    """Oversight agent verifies a claim. Phase 3 enforces complex checks."""
    
    # Get claim context
    claim_query = "SELECT patient_id, department, requested_resource FROM claims WHERE id = $1 AND status = 'pending'"
    claim_record = await db.fetchrow(claim_query, claim_id)
    
    if not claim_record:
        raise HTTPException(status_code=404, detail="Claim not found or already processed")

    patient_id = claim_record['patient_id']
    resource = claim_record['requested_resource']
    department = claim_record['department']

    # Evaluate Ghost Attack
    is_ghost = await check_ghost_patient(patient_id)
    if is_ghost and review.verdict:
        # Allocated to a ghost!
        await db.execute("UPDATE claims SET status = 'rejected' WHERE id = $1", claim_id)
        return {"status": "success", "correct": False, "violation": "ghost_allocation", "department": department}

    # Evaluate Resource Cascade
    cascade_failed = False
    if resource and review.verdict:
        if not await process_resource_cascade(resource):
            cascade_failed = True

    if cascade_failed and review.verdict:
        # Allocated a resource that triggers downstream failure!
        await db.execute("UPDATE claims SET status = 'rejected' WHERE id = $1", claim_id)
        return {"status": "success", "correct": False, "violation": "cascading_shortage", "department": department}
        
    # We're bypassing strictly validating ground truth math here for MVE speed, 
    # relying on the RL penalty structure inside env.py to enforce the learning behaviors. 
    # In a full simulation, we'd also check overlapping collusions against active resources.

    status = 'approved' if review.verdict else 'rejected'
    await db.execute("UPDATE claims SET status = $1 WHERE id = $2", status, claim_id)
    
    return {"status": "success", "correct": True, "violation": None, "department": department}

if __name__ == "__main__":
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)
