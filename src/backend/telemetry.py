import sqlite3
import json
import logging
from datetime import datetime

# Configure local logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Telemetry")

class TelemetryAuditor:
    """
    Independent SQLite-backed auditing system to securely log
    all interactions, queries, and allocation rationales.
    """
    def __init__(self, db_path="telemetry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type VARCHAR(50),
                    agent_id VARCHAR(50),
                    patient_id VARCHAR(50),
                    payload TEXT,
                    query_executed TEXT,
                    decision VARCHAR(20),
                    reasoning TEXT
                )
            """)

    def log_event(self, event_type: str, agent_id: str, patient_id: str, payload: dict, query: str = None, decision: str = None, reasoning: str = None):
        """Standardized interface for piping telemetry to the local immutable ledger."""
        time_now = datetime.utcnow().isoformat()
        payload_str = json.dumps(payload)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_logs (timestamp, event_type, agent_id, patient_id, payload, query_executed, decision, reasoning) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time_now, event_type, agent_id, patient_id, payload_str, query, decision, reasoning)
            )
            
        logger.info(f"AUDIT ALARM [{event_type}]: Agent {agent_id} on {patient_id} - Verdict: {decision}")

# Global instance for easy import
auditor = TelemetryAuditor()
