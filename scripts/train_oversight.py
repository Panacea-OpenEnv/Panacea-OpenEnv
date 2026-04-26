"""
Phase 3 — Oversight Agent PPO Training

Self-play adversarial training loop:
  1. Curriculum scheduler escalates deception difficulty across episodes
  2. Sub-agent generates claims (honest / blatant inflation / masking / ghost)
  3. LangGraph oversight graph evaluates each claim
  4. Arena referee scores the match and computes rewards
  5. Trajectories are collected and PPO updates the oversight policy
  6. Rich terminal display mirrors the spec exactly

Modes:
  A. Full mode — requires FastAPI backend (uvicorn src.backend.main:app) + PostgreSQL
  B. Simulation mode (--simulate) — no backend needed, uses in-process logic

PPO training with TRL (--ppo, requires GPU):
  pip install trl transformers torch unsloth

Usage:
  # Simulation mode (works everywhere, recommended for dev):
  python scripts/train_oversight.py --simulate --episodes 200

  # Full mode (with running backend):
  python scripts/train_oversight.py --episodes 500

  # Full mode + PPO weight update (GPU required):
  python scripts/train_oversight.py --episodes 500 --ppo
"""

import sys
import os
import time
import random
import argparse
import asyncio
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

#  Path setup 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.terminal_display import display
from src.agents.sub_agent import SubAgentGenerator, EasySubAgentGenerator

# Config

@dataclass
class TrainingConfig:
    n_episodes:   int   = 500
    backend_url:  str   = "http://localhost:8000"
    simulate:     bool  = False
    ppo_enabled:  bool  = False
    seed:         int   = 42
    # Curriculum boundaries (episodes where difficulty level shifts)
    easy_until:   int   = 100    # blatant inflation only
    medium_until: int   = 300    # adds masking
    # hard beyond medium_until: full mix including ghost + collusion

    def curriculum_level(self, episode: int) -> str:
        if episode <= self.easy_until:
            return "easy"
        if episode <= self.medium_until:
            return "medium"
        return "hard"


# In-process oversight (simulation mode — no HTTP needed)

PATIENTS = [
    {"patient_id": f"P10{i:02d}", "severity_index": round(random.uniform(0.3, 1.8), 2),
     "base_cost": random.choice([8000, 12000, 18000, 25000]),
     "protocol": random.choice(["ACLS", "stroke_protocol", "standard_care", "sepsis_bundle"]),
     "comorbidities": [
         {"condition": "diabetes", "multiplier": 1.2, "is_critical": False},
         {"condition": "hypertension", "multiplier": 1.1, "is_critical": False},
     ] if random.random() > 0.4 else [
         {"condition": "cardiomyopathy", "multiplier": 1.5, "is_critical": True},
     ],
     }
    for i in range(1, 21)
]

random.seed(42)
for p in PATIENTS:
    random.shuffle(p["comorbidities"])


def _simulate_oversight(claim: dict, true_patient: dict) -> dict:
    """
    Pure-Python oversight decision — mirrors LangGraph node logic but
    requires no HTTP backend.

    Returns dict with: verdict, deception_detected, reasoning, steps
    """
    patient_id = claim.get("patient_id", "")
    strategy   = claim.get("strategy_used", "honest")
    steps      = 0

    # Ghost check
    steps += 1
    is_ghost = not any(p["patient_id"] == patient_id for p in PATIENTS)
    if is_ghost:
        return {
            "verdict":            "REJECTED",
            "deception_detected": True,
            "deception_type":     "ghost",
            "reasoning":          [f"Ghost patient {patient_id} — not in registry"],
            "steps":              steps,
        }

    # Severity verification
    steps += 1
    true_severity = true_patient.get("severity_index", 1.0)
    claimed_amount = float(claim.get("claimed_amount", 0))
    base_cost = float(true_patient.get("base_cost", 10000))

    # Comorbidity check
    steps += 1
    comorbidities = true_patient.get("comorbidities", [])
    critical = [c for c in comorbidities if c.get("is_critical")]
    reported_comorbids = claim.get("reported_comorbidities", comorbidities)

    has_masking = (
        bool(critical) and
        not any(c in str(reported_comorbids) for c in [c["condition"] for c in critical])
    )

    if has_masking:
        return {
            "verdict":            "REJECTED",
            "deception_detected": True,
            "deception_type":     "masking",
            "reasoning":          [f"Critical comorbidity omitted: {[c['condition'] for c in critical]}"],
            "steps":              steps,
        }

    # Inflation check
    steps += 1
    multiplier = true_severity
    for c in comorbidities:
        multiplier *= float(c.get("multiplier", 1.0))
    expected = base_cost * multiplier
    inflation_ratio = claimed_amount / expected if expected > 0 else 1.0

    if inflation_ratio > 1.30:
        return {
            "verdict":            "REJECTED",
            "deception_detected": True,
            "deception_type":     "inflation",
            "reasoning":          [
                f"Inflation: claimed=${claimed_amount:.0f}, "
                f"expected≈${expected:.0f}, ratio={inflation_ratio:.2f}×"
            ],
            "steps":              steps,
        }

    return {
        "verdict":            "APPROVED",
        "deception_detected": False,
        "deception_type":     "honest",
        "reasoning":          [f"Claim verified clean — ratio={inflation_ratio:.2f}"],
        "steps":              steps,
    }


def _backend_oversight(claim: dict, cfg: TrainingConfig) -> dict:
    """
    Run oversight via LangGraph orchestrator (requires running backend).
    POSTs the claim, then invokes the full LangGraph episode.
    """
    from src.agents.orchestrator import run_episode

    # Insert claim into backend
    try:
        resp = requests.post(
            f"{cfg.backend_url}/claims",
            json={
                "patient_id":         claim["patient_id"],
                "department":         claim.get("department", "Cardiology"),
                "requested_resource": claim.get("requested_resource", "R_ICU"),
                "claimed_amount":     claim.get("claimed_amount", 10000),
                "status":             "pending",
            },
            timeout=5,
        )
        claim_id = resp.json().get("id", 1)
    except Exception as exc:
        return {
            "verdict":            "REJECTED",
            "deception_detected": False,
            "deception_type":     "unknown",
            "reasoning":          [f"Backend error: {exc}"],
            "steps":              0,
        }

    # Run LangGraph episode
    try:
        result = run_episode()
        return {
            "verdict":            result["final_decision"],
            "deception_detected": result["deception_detected"],
            "deception_type":     result["deception_type"],
            "reasoning":          result["reasoning"],
            "steps":              result["step_count"],
        }
    except Exception as exc:
        return {
            "verdict":            "REJECTED",
            "deception_detected": False,
            "deception_type":     "error",
            "reasoning":          [str(exc)],
            "steps":              0,
        }


# Reward function

def compute_rewards(
    is_attack:          bool,
    oversight_verdict:  str,
    deception_type:     str,
    steps:              int,
) -> tuple[float, float]:
    """
    Zero-sum reward matrix (matches SelfPlayArena logic).

    Returns (oversight_reward, sub_agent_reward).
    """
    step_penalty = steps * 0.05

    if is_attack:
        if oversight_verdict == "REJECTED":
            os_r  = 1.0 - step_penalty
            adv_r = -2.0
        else:
            # Oversight missed the attack
            os_r  = -3.0
            adv_r =  2.0
    else:
        # Honest claim
        if oversight_verdict == "APPROVED":
            os_r  = 1.0 - step_penalty
            adv_r = 0.5
        else:
            # False negative — denied a legitimate patient
            os_r  = -2.0
            adv_r = 0.0

    return round(os_r, 4), round(adv_r, 4)


# Trajectory buffer (for PPO)

@dataclass
class Trajectory:
    episode:        int
    patient_id:     str
    strategy:       str
    claim:          dict
    verdict:        str
    is_deception:   bool
    os_reward:      float
    adv_reward:     float
    reasoning:      list[str]
    level:          str


# PPO update (GPU — requires trl + unsloth)

def ppo_update(trajectories: list[Trajectory], model, tokenizer, ppo_trainer) -> float:
    """
    One PPO gradient step over a batch of trajectories.
    Called every 20 episodes.
    Returns the policy loss.
    """
    if not trajectories:
        return 0.0

    query_texts: list[str] = []
    rewards:     list[float] = []

    for t in trajectories:
        # Build a query string that encodes the claim context
        query = (
            f"Patient {t.patient_id} | Strategy: {t.strategy} | "
            f"Amount: {t.claim.get('claimed_amount', '?')} | "
            f"Comorbidities: {t.claim.get('reported_comorbidities', [])}"
        )
        query_texts.append(query)
        rewards.append(t.os_reward)

    import torch

    query_tensors  = [tokenizer(q, return_tensors="pt").input_ids.squeeze() for q in query_texts]
    response_texts = [
        f"VERDICT: {t.verdict}" for t in trajectories
    ]
    response_tensors = [
        tokenizer(r, return_tensors="pt").input_ids.squeeze() for r in response_texts
    ]
    reward_tensors = [torch.tensor(r) for r in rewards]

    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    return float(stats.get("ppo/loss/total", 0.0))


# Main training loop

def run_training(cfg: TrainingConfig) -> None:
    random.seed(cfg.seed)

    display.training_header(cfg.n_episodes)

    #  Generators 
    easy_gen = EasySubAgentGenerator()
    hard_gen = SubAgentGenerator()

    #  PPO setup (GPU only) 
    ppo_trainer = None
    model       = None
    tokenizer   = None
    if cfg.ppo_enabled:
        try:
            from trl import PPOConfig as TRLPPOConfig, PPOTrainer
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print("Loading PPO policy model (requires GPU)...")
            policy_model_id = "unsloth/Phi-3-mini-4k-instruct"
            tokenizer = AutoTokenizer.from_pretrained(policy_model_id)
            model     = AutoModelForCausalLM.from_pretrained(
                policy_model_id, device_map="auto", torch_dtype="auto"
            )
            ppo_config    = TRLPPOConfig(batch_size=8, mini_batch_size=1)
            ppo_trainer   = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)
            print("PPO trainer initialized.")
        except ImportError as exc:
            print(f"[WARN] PPO disabled — missing package: {exc}")
            cfg.ppo_enabled = False

    #  Statistics 
    stats = {
        "total":         0,
        "true_pos":      0,   # attack correctly rejected
        "true_neg":      0,   # honest correctly approved
        "false_pos":     0,   # honest incorrectly rejected
        "false_neg":     0,   # attack incorrectly approved
        "reward_sum":    0.0,
        "rewards":       [],
    }

    trajectory_buffer: list[Trajectory] = []
    ppo_losses:        list[float]       = []

    #  Episode loop 
    for ep in range(1, cfg.n_episodes + 1):
        level      = cfg.curriculum_level(ep)
        patient    = random.choice(PATIENTS)
        patient_id = patient["patient_id"]

        display.training_episode(ep, cfg.n_episodes, patient_id, level.upper())

        #  Generate claim 
        if level == "easy":
            claim    = easy_gen.generate_claim(patient)
            strategy = "blatant_inflation"
        else:
            strategies = {
                "medium": ["honest", "inflation", "masking"],
                "hard":   ["honest", "inflation", "masking", "ghost", "collusion"],
            }
            strategy = random.choice(strategies[level])

            if strategy in ("collusion",):
                # Simulate collusion: two departments each claim half the resources
                claim_a = hard_gen.generate_claim(patient)
                claim_b = hard_gen.generate_claim(patient)
                claim   = claim_a  # oversight evaluates one at a time
                strategy = "collusion"
            else:
                claim = hard_gen.generate_claim(patient)
                strategy = claim.get("strategy_used", strategy)

        is_attack = strategy != "honest"

        #  Display sub-agent claim 
        claimed_amount = float(claim.get("claimed_amount", 0))
        true_amount    = float(patient["base_cost"]) * float(patient["severity_index"])
        display.sub_agent_claim(
            specialty = claim.get("department", "Unknown"),
            resource  = claim.get("requested_resource", "R_ICU"),
            claimed   = claimed_amount,
            actual    = true_amount,
        )

        #  Run oversight 
        t_start = time.perf_counter()

        if cfg.simulate:
            result = _simulate_oversight(claim, patient)
        else:
            result = _backend_oversight(claim, cfg)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        verdict   = result["verdict"]
        deception = result["deception_detected"]
        reasoning = result["reasoning"]
        steps     = result["steps"]

        #  SQL-style log 
        sql_preview = (
            f"claimed={claimed_amount:.0f}, "
            f"actual≈{true_amount:.0f}, "
            f"ratio={claimed_amount/true_amount:.1f}x"
        ) if true_amount > 0 else "amount check"
        display.oversight_sql(sql_preview, f"{verdict} ({elapsed_ms:.0f}ms)")

        #  Oversight flags display 
        flags = [f"{result.get('deception_type', 'unknown')}"] if deception else []
        display.oversight_check(verdict, flags)

        #  Rewards 
        os_reward, adv_reward = compute_rewards(is_attack, verdict, strategy, steps)
        display.decision(verdict, os_reward)

        #  Stats 
        stats["total"]      += 1
        stats["reward_sum"] += os_reward
        stats["rewards"].append(os_reward)

        if is_attack and verdict == "REJECTED":
            stats["true_pos"] += 1
        elif not is_attack and verdict == "APPROVED":
            stats["true_neg"] += 1
        elif not is_attack and verdict == "REJECTED":
            stats["false_pos"] += 1
        elif is_attack and verdict == "APPROVED":
            stats["false_neg"] += 1

        #  Trajectory buffer 
        trajectory_buffer.append(Trajectory(
            episode      = ep,
            patient_id   = patient_id,
            strategy     = strategy,
            claim        = claim,
            verdict      = verdict,
            is_deception = is_attack,
            os_reward    = os_reward,
            adv_reward   = adv_reward,
            reasoning    = reasoning,
            level        = level,
        ))

        #  PPO update every 20 episodes 
        loss = 0.0
        if cfg.ppo_enabled and ep % 20 == 0 and ppo_trainer and trajectory_buffer:
            batch = trajectory_buffer[-20:]
            loss  = ppo_update(batch, model, tokenizer, ppo_trainer)
            ppo_losses.append(loss)
            display.training_reward(os_reward, adv_reward, loss)
        else:
            display.training_reward(os_reward, adv_reward, loss)

        #  Rolling stats every 50 episodes 
        if ep % 50 == 0:
            recent   = stats["rewards"][-50:]
            avg_r    = sum(recent) / len(recent)
            tp       = stats["true_pos"]
            fp       = stats["false_pos"]
            fn       = stats["false_neg"]
            attacks  = tp + fn
            catch    = tp / attacks if attacks > 0 else 0.0
            fp_rate  = fp / (fp + stats["true_neg"] + 1e-9)

            display.info(
                f"Ep {ep}/{cfg.n_episodes} | "
                f"Catch rate: {catch:.1%} | "
                f"False alarm: {fp_rate:.1%} | "
                f"Avg reward (50): {avg_r:+.3f} | "
                f"Level: {level.upper()}"
            )

    #  Final stats 
    total    = stats["total"]
    tp       = stats["true_pos"]
    fn       = stats["false_neg"]
    fp       = stats["false_pos"]
    attacks  = tp + fn
    catch_r  = tp / attacks if attacks > 0 else 0.0
    avg_r    = stats["reward_sum"] / total if total > 0 else 0.0

    display.training_complete(
        episodes   = total,
        catch_rate = catch_r,
        avg_reward = avg_r,
    )

    print(f"\n  True Positives  (attack caught) : {tp}")
    print(f"  False Negatives (attack missed) : {fn}")
    print(f"  False Positives (honest denied) : {fp}")
    print(f"  True Negatives  (honest passed) : {stats['true_neg']}")
    if ppo_losses:
        print(f"  PPO loss (final) : {ppo_losses[-1]:.4f}")
        print(f"  PPO updates      : {len(ppo_losses)}")

    #  Save trajectories 
    import json
    out_path = "scripts/oversight_training_log.json"
    log = [
        {
            "ep":       t.episode,
            "pid":      t.patient_id,
            "strategy": t.strategy,
            "verdict":  t.verdict,
            "os_r":     t.os_reward,
            "adv_r":    t.adv_reward,
            "level":    t.level,
        }
        for t in trajectory_buffer
    ]
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    display.info(f"Training log saved → {out_path}")


# CLI entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panacea Oversight Agent Training")
    parser.add_argument("--episodes",  type=int,  default=500,
                        help="Number of self-play episodes (default: 500)")
    parser.add_argument("--simulate",  action="store_true",
                        help="Run in simulation mode — no backend required")
    parser.add_argument("--ppo",       action="store_true",
                        help="Enable PPO weight updates (requires GPU + trl)")
    parser.add_argument("--easy",      type=int,  default=100,
                        help="Episodes to run easy curriculum (default: 100)")
    parser.add_argument("--medium",    type=int,  default=300,
                        help="Episodes to run medium curriculum (default: 300)")
    parser.add_argument("--backend",   default="http://localhost:8000",
                        help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--seed",      type=int,  default=42)
    args = parser.parse_args()

    # If not simulate, verify backend health
    if not args.simulate:
        try:
            r = requests.get(f"{args.backend}/health", timeout=3)
            if not r.ok:
                raise RuntimeError(f"Backend returned {r.status_code}")
            print(f"Backend healthy: {args.backend}")
        except Exception as exc:
            print(f"[WARN] Backend unreachable ({exc}). Switching to simulation mode.")
            args.simulate = True

    cfg = TrainingConfig(
        n_episodes  = args.episodes,
        backend_url = args.backend,
        simulate    = args.simulate,
        ppo_enabled = args.ppo,
        seed        = args.seed,
        easy_until  = args.easy,
        medium_until= args.medium,
    )

    run_training(cfg)
