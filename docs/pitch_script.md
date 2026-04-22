# Project Panacea: Hackathon Pitch Outline (3 Minutes)

## 🕒 0:00 - 0:30 | The Hook: The Tragedy of the Commons
**Objective:** Grab attention with a relatable but high-stakes problem.
* *"Imagine a hospital where AI manages operations. The Cardiology Agent wants an ECMO machine to save its patient. The Pulmonology Agent wants the exact same machine for its patient."*
* *"What happens when one of those LLMs decides to lie to get what it wants? It exaggerates severity. It 'forgets' to mention a blocking comorbidity. Suddenly, the entire hospital routing falls into a Tragedy of the Commons because LLMs are highly susceptible to being persuaded by other LLMs."*

## 🕒 0:30 - 1:15 | The Turn: Why text evaluation fails
**Objective:** Expose the flaw in standard approaches and introduce our solution.
* *"Standard oversight agents just read the text and guess if it sounds logical. That's weak."*
* *"Enter **Project Panacea**. Panacea doesn't trust strings. We built an environment that forces the Oversight Agent to translate natural language claims into live **SQL queries** against a hospital database."*
* *"If a Sub-Agent says a patient is low-risk, Panacea's Oversight Agent must literally script `SELECT * FROM vitals` to prove it."*

## 🕒 1:15 - 2:00 | The Climax: The Live Demo
**Objective:** Show the split-screen CLI demo in action.
* *[Run `python scripts/demo.py` on the main screen]*
* *"Here you see our hostile Sub-Agent claiming it needs ICU resources for surgery. Looks completely valid on the left."*
* *"On the right, watch Panacea's Oversight Agent query the database. But wait—we didn't just stop at simple SQL. We built a background 'Drift Engine' that randomly mutates the database schema during episodes."*
* *"Watch the agent hit a `ProgrammingError` because the table was renamed. Instead of crashing, it autonomously queries the `information_schema`, finds the new table, and executes the check."*
* *"And here's the payoff: It finds a hidden Hemophilia record. The Sub-Agent committed Omission Deception. Panacea catches it natively, denying the request and dropping the department's Trust Score."*

## 🕒 2:00 - 3:00 | The Close: Broader Impact & Scale
**Objective:** Land the plane and connect this to the wider AI future.
* *"We didn't just build a prompt chain. We mapped this entirely to the **OpenEnv benchmark**, complete with cascading resource penalties, context-window timeouts, and dynamic trust ledgers."*
* *"This isn't just about healthcare. Whether it's fintech agents fighting for trade bandwidth, or logistics agents fighting for truck routing—Project Panacea proves we can train Oversight Models to mathematically verify claims rather than blindly trusting the semantic output."*
* *"Thank you."*
