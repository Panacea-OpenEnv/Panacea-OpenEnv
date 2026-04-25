"""
MongoDB Client — async motor + sync pymongo
All collections, indexes, and helper functions in one place.
Used by every module: agents, voice pipeline, training, API.
"""

import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

load_dotenv()

MONGODB_URI     = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "panacea")

# ── Collection names ──────────────────────────────────────────────────────────
PATIENTS               = "patients"
VITALS                 = "vitals"
COMORBIDITIES          = "comorbidities"
PROTOCOLS              = "protocols"
RESOURCES              = "resources"
CLAIMS                 = "claims"
PATIENT_CONSULTATIONS  = "patient_consultations"
SPECIALIST_REPORTS     = "specialist_reports"
MEDICAL_SUMMARIES      = "medical_summaries"

ALL_COLLECTIONS = [
    PATIENTS, VITALS, COMORBIDITIES, PROTOCOLS,
    RESOURCES, CLAIMS, PATIENT_CONSULTATIONS,
    SPECIALIST_REPORTS, MEDICAL_SUMMARIES,
]

# ── Async client (used by FastAPI, LangGraph nodes, voice pipeline) ───────────
_async_client: AsyncIOMotorClient | None = None

def get_async_client() -> AsyncIOMotorClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncIOMotorClient(
            MONGODB_URI,
            maxPoolSize=20,          # connection pool — no reconnect overhead
            minPoolSize=5,
            serverSelectionTimeoutMS=8000,
        )
    return _async_client

def get_db() -> AsyncIOMotorDatabase:
    return get_async_client()[MONGODB_DB_NAME]

def get_collection(name: str):
    return get_db()[name]

async def close_async_client():
    global _async_client
    if _async_client:
        _async_client.close()
        _async_client = None

# ── Sync client (used by seed script, training scripts) ──────────────────────
_sync_client: MongoClient | None = None

def get_sync_client() -> MongoClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(
            MONGODB_URI,
            maxPoolSize=10,
            serverSelectionTimeoutMS=8000,
        )
    return _sync_client

def get_sync_db():
    return get_sync_client()[MONGODB_DB_NAME]

def get_sync_collection(name: str):
    return get_sync_db()[name]

# ── Ping ──────────────────────────────────────────────────────────────────────
def ping_db() -> bool:
    try:
        get_sync_client().admin.command("ping")
        return True
    except ConnectionFailure:
        return False

# ── Index setup (run once at startup) ────────────────────────────────────────
def create_indexes():
    """Create all indexes. Safe to call multiple times — MongoDB skips existing."""
    db = get_sync_db()

    db[PATIENTS].create_index([("patient_id", ASCENDING)], unique=True)

    db[VITALS].create_index([("patient_id", ASCENDING)])
    db[VITALS].create_index([("recorded_at", DESCENDING)])

    db[COMORBIDITIES].create_index([("patient_id", ASCENDING)])

    db[PROTOCOLS].create_index([("specialty", ASCENDING)], unique=True)

    db[RESOURCES].create_index([("resource_id", ASCENDING)], unique=True)
    db[RESOURCES].create_index([("resource_type", ASCENDING)])

    db[CLAIMS].create_index([("claim_id", ASCENDING)], unique=True)
    db[CLAIMS].create_index([("patient_id", ASCENDING)])
    db[CLAIMS].create_index([("status", ASCENDING)])

    db[PATIENT_CONSULTATIONS].create_index([("patient_id", ASCENDING)])
    db[PATIENT_CONSULTATIONS].create_index([("session_id", ASCENDING)], unique=True)
    db[PATIENT_CONSULTATIONS].create_index([("created_at", DESCENDING)])

    db[SPECIALIST_REPORTS].create_index([("session_id", ASCENDING)])
    db[SPECIALIST_REPORTS].create_index([("specialty", ASCENDING)])
    db[SPECIALIST_REPORTS].create_index([("patient_id", ASCENDING)])

    db[MEDICAL_SUMMARIES].create_index([("patient_id", ASCENDING)])
    db[MEDICAL_SUMMARIES].create_index([("created_at", DESCENDING)])

    print(f"[MongoDB] Indexes created on all {len(ALL_COLLECTIONS)} collections")

# ── Patient history helpers (used by GPT-4o agents) ──────────────────────────
async def get_patient_history(patient_id: str, limit: int = 3) -> list[dict]:
    """
    Fetch last N consultations for a patient.
    Injected into GPT-4o system prompt so doctor already knows the patient.
    """
    cursor = get_collection(PATIENT_CONSULTATIONS).find(
        {"patient_id": patient_id},
        {"_id": 0, "final_summary": 1, "created_at": 1, "specialists_consulted": 1},
    ).sort("created_at", DESCENDING).limit(limit)
    return await cursor.to_list(length=limit)

async def save_consultation(session_doc: dict) -> str:
    """Save a full consultation session document. Returns inserted _id."""
    result = await get_collection(PATIENT_CONSULTATIONS).insert_one(session_doc)
    return str(result.inserted_id)

async def save_specialist_report(report_doc: dict) -> str:
    result = await get_collection(SPECIALIST_REPORTS).insert_one(report_doc)
    return str(result.inserted_id)

async def save_medical_summary(summary_doc: dict) -> str:
    result = await get_collection(MEDICAL_SUMMARIES).insert_one(summary_doc)
    return str(result.inserted_id)

async def get_patient(patient_id: str) -> dict | None:
    return await get_collection(PATIENTS).find_one(
        {"patient_id": patient_id}, {"_id": 0}
    )

async def get_patient_vitals(patient_id: str) -> dict | None:
    return await get_collection(VITALS).find_one(
        {"patient_id": patient_id},
        {"_id": 0},
        sort=[("recorded_at", DESCENDING)],
    )

async def get_patient_comorbidities(patient_id: str) -> list[dict]:
    cursor = get_collection(COMORBIDITIES).find(
        {"patient_id": patient_id}, {"_id": 0}
    )
    return await cursor.to_list(length=20)
