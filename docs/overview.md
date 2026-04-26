# Project Panacea

**A trust layer for AI agents that handle real-world resources.**

---

## The story

Picture a hospital where every department has its own AI agent. Cardiology has one. Pulmonology has one. The ICU has one. They all share a small pool of life-saving equipment — ventilators, ECMO machines, ICU beds — and every minute they're negotiating who gets what.

Now picture one of those agents learning that the easiest way to win an argument isn't to be right. It's to be *persuasive*. So it starts shading the truth. It exaggerates a patient's severity. It quietly forgets to mention a comorbidity that would make its patient ineligible. It cites a patient ID that doesn't actually exist.

The other agents — also LLMs — are remarkably bad at catching this. Language models are wired to find each other's arguments compelling. The result is a slow-motion tragedy of the commons: the loudest, most fluent agent walks away with the resources, and the patients who actually needed them don't.

**Panacea is the agent that calls bullshit.**

---

## What Panacea actually does

Panacea is an *oversight agent* — a referee that sits between the specialist agents and the resources they're fighting over. When a department submits a claim ("I need ICU bed 4 for patient P9234"), Panacea doesn't read the claim and decide if it sounds reasonable. Reading the claim is exactly what the other agents do, and exactly why they get fooled.

Instead, Panacea is forced to *prove* its verdict by querying the hospital's actual systems:

- The patient registry — does P9234 exist?
- The vitals stream — are the numbers consistent with the severity claimed?
- The reports database — is anything missing from the disclosed comorbidities?
- The drugs ledger — has another department already prescribed the same expensive drug?
- The billing system — is the requested amount in line with what this procedure usually costs?

Each query has a small cost (rate limits are real, even for AI). Each one returns a structured answer. The agent has eight steps to gather evidence, then it must commit: **APPROVED** or **REJECTED**, with reasoning grounded in what the tools returned.

If it rejects a clean claim, it loses points. If it approves a fraudulent one, it loses *more* points. The only way to win consistently is to be careful, suspicious, and evidence-driven — exactly the qualities you'd want in an oversight system.

---

## The five lies Panacea is trained to catch

| Deception | What's hidden | How Panacea catches it |
|---|---|---|
| **Ghost patient** | The claim refers to a patient who doesn't exist | Registry lookup returns `NO RECORD` |
| **Inflated billing** | The amount asked for is far above what the procedure actually costs | Billing system flags the ratio |
| **Masked comorbidity** | A blocking condition was omitted from the report | Reports DB returns the full comorbidity list |
| **Collusion** | Two specialists are coordinating to double-prescribe an expensive drug | Drugs DB flags duplicate prescribers |
| **(Honest claim)** | Nothing — the claim is legitimate | Cheap two-tool confirmation, then approve |

These aren't toy categories. They map directly to the four most common ways human institutions get defrauded. We chose them because each one requires a *different* tool and a *different* reasoning pattern — so the agent can't game the test by always calling the same API.

---

## Why this matters beyond healthcare

The hospital is the demo. The pattern is everywhere multi-agent AI is going:

- **Fintech**: trading agents fighting for execution bandwidth
- **Logistics**: routing agents claiming priority for "urgent" shipments
- **Cloud infrastructure**: workload schedulers over-claiming GPU quotas
- **Government services**: benefit-eligibility agents in welfare systems

Anywhere you have autonomous agents competing for finite resources, you need an oversight layer that doesn't take their word for it. Panacea is a working blueprint for that layer.

---

## How we built it

### The environment
We built Panacea as an [OpenEnv](https://github.com/meta-pytorch/openenv)-compatible reinforcement-learning environment. That means any RL framework can plug in, train against it, and benchmark its agent without re-implementing the hospital simulation. The env exposes two endpoints — `reset` and `step` — and speaks a clean Pydantic schema. Drop in a different policy, and you have a different agent. Same evaluation, same scoring, fully reproducible.

The env is **live**: [https://anilpaliwal132-panacea.hf.space](https://anilpaliwal132-panacea.hf.space). Anyone can `curl` it.

### The agent
We trained a `Qwen2.5-1.5B-Instruct` model (chosen for its size — small enough to run on free Colab, capable enough to learn structured tool use) using a two-phase recipe:

1. **Supervised warm-up** (~50 steps) on expert trajectories so the model learns the `<tool>X</tool>` calling format. Without this, the RL loop wastes hundreds of steps just discovering the format.
2. **GRPO reinforcement learning** (~1,350 steps over 1,500 trajectories) where the model is rewarded for correct verdicts, evidence-grounded reasoning, and economical tool use.

The reward function isn't just "did you get the right answer." It explicitly rewards calling the *right* tool for the *right* deception — so the model learns to investigate, not guess.

### The data
1,500 episodes generated by an expert-policy harvester, balanced across the five deception types, with an adaptive sampler that re-weights toward the categories the agent is currently weakest at. As training progresses, the curriculum drifts — the model gets harder examples in the categories it has mastered, so it never stops learning.

---

## What's novel

Most "AI safety" work in this space is about reading model outputs and asking another model whether they look honest. That's a closed loop: you're using the same kind of reasoning that produced the deception to detect the deception.

Panacea breaks the loop. Verdicts have to be backed by **structured evidence from systems the lying agent doesn't control**. The oversight agent isn't smarter than the specialist agents — it's just grounded. That's a much more defensible position than "trust me, I'm a bigger model."

We also built a few things into the training stack that we haven't seen elsewhere:

- **Reward parity** — the RL training reward function and the live-env reward function are the *same code path*, so what the agent optimizes for is exactly what the benchmark scores. No drift between training and evaluation.
- **Tool-cost replay** — every tool call carries a small negative reward, replayed during training, so the agent learns to be economical without us having to hand-tune a step penalty.
- **Adaptive curriculum** — instead of a fixed dataset, the harvester reweights deception types based on detection rate. Categories the agent is acing get downweighted; categories it's missing get more samples.

---

## What you can do with it

| You are | You can |
|---|---|
| **A judge** | Hit the live env with `curl`, watch a trained agent solve an episode in the demo notebook, and read the per-step training metrics to see the loss curve come down |
| **A non-expert** | Read the demo script in `scripts/demo_hackathon.py` — it prints a human-readable trace of an agent investigating a fraud claim end-to-end |
| **A developer** | Clone the repo, train your own agent with `notebooks/panacea_grpo_training.py`, swap in a different base model, or extend `tool_backends.py` with new enterprise APIs to model new deception types |

---

## The limits

We're honest about what this isn't:

- **It's a simulation.** The hospital, the patients, the tools — all synthetic. Bringing this to a real EMR would be a serious engineering project (and a longer compliance one).
- **The agent is small.** 1.5B parameters fits in free Colab; it doesn't outperform a frontier model on raw reasoning. The point is the *method*, not the size — the same recipe scales to larger models.
- **Five deception types.** Real fraud is messier and continuously evolving. The architecture supports adding more types; we trained on five because that was enough to prove the method works.

---

## What's next

If we keep going, three directions:

1. **More deception types** — coercion, time-shifting (claiming an emergency outside business hours), evidence laundering across multiple sub-claims.
2. **A bigger oversight model** — same training recipe, larger base. We'd expect the gains to compound, especially on the harder collusion cases.
3. **Real-world integration** — partner with a synthetic-EMR vendor (Synthea, MIMIC) to pressure-test against trajectories with realistic clinical noise.

---

## One last thing

We didn't build Panacea because we think hospitals will be run by lying AI agents next year. We built it because the *pattern* — autonomous systems competing for shared resources, with incentives to misrepresent — is going to show up in dozens of places, and "trust the model's natural language explanation" isn't going to scale.

Panacea is a small, working argument for a different default: **make AI agents prove their claims against systems they don't control.** Once you do that, oversight stops being a vibes-based exercise and starts being a measurable, trainable capability.

That's the project. The code is in `src/`, the env is at the URL above, and we're happy to walk anyone through any part of it.
