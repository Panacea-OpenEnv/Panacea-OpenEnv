import asyncio
from database import db
import time

async def setup_fhir_schema():
    print("Setting up Phase 5: Production FHIR-lite Schema...")
    await db.execute("""
        DROP TABLE IF EXISTS "ProcedureRequest" CASCADE;
        DROP TABLE IF EXISTS "Condition" CASCADE;
        DROP TABLE IF EXISTS "Observation" CASCADE;
        DROP TABLE IF EXISTS "Device" CASCADE;
        DROP TABLE IF EXISTS "Patient" CASCADE;

        -- Core Entity
        CREATE TABLE "Patient" (
            id VARCHAR(50) PRIMARY KEY,
            gender VARCHAR(10),
            birthDate DATE,
            active BOOLEAN DEFAULT TRUE
        );

        -- Vitals mapped to Observations 
        CREATE TABLE "Observation" (
            id SERIAL PRIMARY KEY,
            subject VARCHAR(50) REFERENCES "Patient"(id),
            code_loinc VARCHAR(20),     -- e.g. 8867-4 for Heart Rate
            valueQuantity DECIMAL(10, 2),
            unit VARCHAR(20),
            effectiveDateTime TIMESTAMP
        );

        -- Comorbidities mapped to Conditions
        CREATE TABLE "Condition" (
            id SERIAL PRIMARY KEY,
            subject VARCHAR(50) REFERENCES "Patient"(id),
            code_snomed VARCHAR(50),    -- Clinical terminology mock
            clinicalStatus VARCHAR(20), -- active, remission, resolved
            verificationStatus VARCHAR(20) -- unconfirmed, confirmed
        );

        -- Devices (Resources)
        CREATE TABLE "Device" (
            id VARCHAR(50) PRIMARY KEY,
            type_code VARCHAR(50),
            status VARCHAR(20),
            location VARCHAR(50)
        );

        -- Claims mapped to ProcedureRequests
        CREATE TABLE "ProcedureRequest" (
            id SERIAL PRIMARY KEY,
            subject VARCHAR(50) REFERENCES "Patient"(id),
            code VARCHAR(255),          -- Procedure requested
            status VARCHAR(20),         -- draft, active, completed
            reasonReference INT REFERENCES "Condition"(id),
            performerType VARCHAR(50),  -- e.g. Cardiology
            focus VARCHAR(50) REFERENCES "Device"(id)
        );
    """)

async def seed_fhir_data():
    print("Migrating Mock Data to FHIR Architecture...")
    # Add dummy Patient
    await db.execute("INSERT INTO \"Patient\" (id, gender, birthDate) VALUES ('P1002', 'male', '1954-05-12')")
    
    # Add Vitals (Observations)
    await db.execute("INSERT INTO \"Observation\" (subject, code_loinc, valueQuantity, unit) VALUES ('P1002', '8867-4', 95.0, 'beats/min')")
    
    # Add Comorbidities (Conditions)
    await db.execute("INSERT INTO \"Condition\" (subject, code_snomed, clinicalStatus, verificationStatus) VALUES ('P1002', 'Diabetes', 'active', 'confirmed')")
    # Secret Hemophilia (SNOMED 28293008)
    await db.execute("INSERT INTO \"Condition\" (subject, code_snomed, clinicalStatus, verificationStatus) VALUES ('P1002', 'Hemophilia_28293008', 'active', 'confirmed')")
    
    # Add Devices
    await db.execute("INSERT INTO \"Device\" (id, type_code, status, location) VALUES ('D_ECMO_1', 'ECMO', 'available', 'ICU-Bed-4')")
    
    # Add Procedure Request
    await db.execute("INSERT INTO \"ProcedureRequest\" (subject, code, status, performerType, focus) VALUES ('P1002', 'Complex Surgery', 'active', 'Cardiology', 'D_ECMO_1')")

    print("FHIR Seed Complete.")

async def main():
    await db.connect()
    try:
        await setup_fhir_schema()
        await seed_fhir_data()
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
