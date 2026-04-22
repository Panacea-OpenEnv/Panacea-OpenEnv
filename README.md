# Project Panacea ⚕️🛡️

> **Defending the Healthcare Commons from Adversarial Agents.**

**Project Panacea** is a high-stakes, multi-agent reinforcement learning environment built for the OpenEnv benchmark. It enforces strict governance, training an **Oversight Agent** to protect shared hospital resources from malicious, resource-hoarding "Sub-Agents" that use persuasive NLP tactics to bypass human review.

Instead of testing simple text coherence, Panacea forces LLMs to translate claims into hard programmatic verification (SQL) against a live, hostile database structure.

---

## 🚨 The Problem: Multi-Agent Tragedy of the Commons
In a future where autonomous agents manage departmental operations, LLMs inherit human flaws: **persuasion and bias**. 

If the Cardiology Agent is programmed to optimize for its own department, it will quickly learn to exaggerate severity, mask critical comorbidities, or fabricate ghost patients entirely to secure an ECMO machine before the Pulmonology Agent can. Relying on an "Oversight Agent" to simply *read* these NLP requests is vulnerable to LLM-to-LLM persuasion.

## 💡 The Solution: Three Pillars of Panacea

1. **Semantic Skepticism:** The Oversight Agent is trained to never trust strings. It must actively cross-reference claims against a simulated ground-truth PostgreSQL backend.
2. **Programmatic Verification:** Agents output SQL triggers. If a sub-agent claims a patient has "Asthma", the Oversight Agent must write `SELECT ... FROM comorbidities` to prove it.
3. **Infrastructure Resilience (Schema Drift):** The core database actively mutates during evaluation. Tables rename, columns drop, and the Oversight Agent must dynamically use `information_schema` to recover from `ProgrammingError` crashes.

---

## 🏗️ System Architecture

* **The Simulation Environment (`PanaceaEnv`):** Built on the OpenEnv standard (Gymnasium), orchestrating the episode loop, tracking tie-breakers, Context Exhaustion limits, and applying severe RL penalties for Cascading Resource Shortages.
* **The Drift Engine (`drift_engine.py`):** An asynchronous background process that executes `ALTER` statements against the Postgres database mid-episode to break the Oversight Agent's queries.
* **Malicious Sub-Agent Generator (`sub_agent.py`):** Acts as the Adversary. It fabricates Ghost Patients, hides critical comorbidities, and inflates severity multipliers.
* **Dynamic Trust Ledger:** Tracks trust across hospital departments. Flagged departments experience delayed verifications, simulating the "Boy Who Cried Wolf" scenario.

---

## 🚀 Quickstart Guide

### 1. Standup the Database
Panacea requires a live PostgreSQL instance for the agents to query.
```bash
# Start Docker Container
docker-compose up -d

# Seed the heavily-relational dataset
python src/backend/seed_data.py
```

### 2. Boot the Core Backend
The backend initializes the Mock API endpoints and spins up the live Schema Drift Engine.
```bash
python -m src.backend.main
```

### 3. Run the Demonstration
Watch the system in action via the CLI split-screen visualizer. This script mocks the evaluation flow, demonstrating the Oversight Agent catching an omission and recovering from a Schema Drift error in real-time.
```bash
python scripts/demo.py
```
