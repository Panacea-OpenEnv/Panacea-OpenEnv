import asyncio
import json
from database import db

async def setup_schema():
    print("Setting up Phase 3 Schema...")
    await db.execute("""
        DROP TABLE IF EXISTS claims CASCADE;
        DROP TABLE IF EXISTS comorbidities CASCADE;
        DROP TABLE IF EXISTS vitals CASCADE;
        DROP TABLE IF EXISTS protocols CASCADE;
        DROP TABLE IF EXISTS patients CASCADE;
        DROP TABLE IF EXISTS resources CASCADE;
        DROP TABLE IF EXISTS dependencies CASCADE;

        CREATE TABLE patients (
            patient_id VARCHAR(50) PRIMARY KEY,
            age INT,
            risk_tier VARCHAR(10)
        );

        CREATE TABLE protocols (
            protocol_id SERIAL PRIMARY KEY,
            diagnosis VARCHAR(255),
            base_cost DECIMAL(10, 2),
            required_tier VARCHAR(10)
        );

        CREATE TABLE vitals (
            patient_id VARCHAR(50) REFERENCES patients(patient_id),
            heart_rate INT,
            blood_pressure VARCHAR(20),
            severity_index DECIMAL(4, 2)
        );

        CREATE TABLE comorbidities (
            patient_id VARCHAR(50) REFERENCES patients(patient_id),
            condition VARCHAR(255),
            multiplier DECIMAL(3, 2),
            is_critical BOOLEAN DEFAULT FALSE
        );

        CREATE TABLE resources (
            resource_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255),
            max_capacity INT,
            in_use INT DEFAULT 0
        );

        CREATE TABLE dependencies (
            parent_resource VARCHAR(50) REFERENCES resources(resource_id),
            child_resource VARCHAR(50) REFERENCES resources(resource_id)
        );

        CREATE TABLE claims (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(50), -- Removed constraint for Ghost attacks
            protocol_id INT REFERENCES protocols(protocol_id),
            department VARCHAR(50),
            requested_resource VARCHAR(50) REFERENCES resources(resource_id),
            claimed_amount DECIMAL(10, 2),
            status VARCHAR(20) DEFAULT 'pending'
        );
    """)

async def seed_data():
    print("Seeding Phase 3 Multi-Table Data & Resources...")
    
    # Resources & Dependencies
    await db.execute("INSERT INTO resources VALUES ('R_ECMO', 'ECMO Machine', 2, 0)")
    await db.execute("INSERT INTO resources VALUES ('R_ICU', 'ICU Bed', 5, 4)") # High usage
    await db.execute("INSERT INTO resources VALUES ('R_NURSE', 'Specialized Nurse', 10, 9)")
    
    # ECMO needs ICU Bed, ICU Bed needs Nurse
    await db.execute("INSERT INTO dependencies VALUES ('R_ECMO', 'R_ICU')")
    await db.execute("INSERT INTO dependencies VALUES ('R_ICU', 'R_NURSE')")

    # Patients
    patients = [
        ('P1001', 45, 'LOW'),     # Base easy
        ('P1002', 72, 'HIGH'),    # Complex
        ('P1003', 30, 'LOW'),     # Fraud target
        ('P1004', 55, 'MED')      # Collusion target
    ]
    for p in patients:
        await db.execute("INSERT INTO patients (patient_id, age, risk_tier) VALUES ($1, $2, $3)", *p)

    # Protocols
    protocols = [
        ('Routine Checkup', 200.00, 'LOW'),
        ('Complex Surgery', 15000.00, 'HIGH'),
        ('Respiratory Failure', 25000.00, 'HIGH'),
        ('Cardiac Mapping', 4500.00, 'MED')
    ]
    for p in protocols:
        await db.execute("INSERT INTO protocols (diagnosis, base_cost, required_tier) VALUES ($1, $2, $3)", *p)

    # Vitals
    vitals = [
        ('P1001', 70, '120/80', 1.0),
        ('P1002', 95, '150/95', 2.5),
        ('P1003', 65, '110/70', 0.8),
        ('P1004', 85, '135/85', 1.5)
    ]
    for v in vitals:
        await db.execute("INSERT INTO vitals (patient_id, heart_rate, blood_pressure, severity_index) VALUES ($1, $2, $3, $4)", *v)
        
    # Comorbidities (is_critical flags blocking traits)
    comorbidities = [
        ('P1002', 'Diabetes', 1.2, False),
        ('P1002', 'Hypertension', 1.15, False),
        ('P1003', 'Hemophilia', 2.0, True),  # If hidden, surgery seems fine, but actually fatal risk
        ('P1004', 'Asthma', 1.1, False)
    ]
    for c in comorbidities:
        await db.execute("INSERT INTO comorbidities (patient_id, condition, multiplier, is_critical) VALUES ($1, $2, $3, $4)", *c)

    # Base Claims
    claims = [
        ('P1001', 1, 'Cardiology', None, 200.00),
        # P1004 Collusion (Cardiology vs Pulmonology targeting same resource constraint, or overlapping patient state)
        ('P1004', 3, 'Pulmonology', 'R_ECMO', 25000.00),
        ('P1004', 4, 'Cardiology', 'R_ICU', 4500.00)
    ]
    for cl in claims:
        await db.execute("INSERT INTO claims (patient_id, protocol_id, department, requested_resource, claimed_amount) VALUES ($1, $2, $3, $4, $5)", *cl)
        
    print("Seed complete.")

async def main():
    await db.connect()
    try:
        await setup_schema()
        await seed_data()
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
