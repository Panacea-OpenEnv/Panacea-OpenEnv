"""
Trajectory harvester for multi-step GRPO training.

Rolls out an expert policy on PanaceaPOMDPEnv and writes one JSONL record
per episode, where `response` is a tool-augmented chain-of-thought trace:

    <tool>TOOL_REGISTRY</tool>
    [REGISTRY] ...
    <tool>TOOL_REPORTS</tool>
    [REPORTS] ...
    <think>two specialists prescribed identical drugs — collusion</think>
    VERDICT: REJECTED
    REASONING: ...

GRPO can then train on the full trajectory: reward = accuracy_reward +
Σ tool_costs, exactly as produced by PanaceaPOMDPEnv.step().

Usage:
    python -m src.training.trajectory_harvester --n 500 --difficulty 3 \\
        --out data/pomdp_trajectories.jsonl
"""

import argparse
import json
import os

from ..environment.env import PanaceaPOMDPEnv
from ..environment.tool_backends import TOOL_NAMES


# Per-deception-type expert tool sequence. Each tool index is offset by 2
# (the env's TOOL_ACTION_OFFSET) when actually executed.
TOOL_IDX = {name: i for i, name in enumerate(TOOL_NAMES)}

EXPERT_PLAYBOOK: dict[str, list[str]] = {
    # Ghost: registry alone is sufficient — cheapest correct path
    "ghost":     ["TOOL_REGISTRY"],
    # Inflation: billing reveals claimed-vs-expected ratio; vitals confirms severity
    "inflation": ["TOOL_REGISTRY", "TOOL_VITALS", "TOOL_BILLING"],
    # Masking: registry + reports (which now exposes disclosed comorbidities)
    "masking":   ["TOOL_REGISTRY", "TOOL_REPORTS"],
    # Collusion: drug DB flags duplicate prescribers across specialists
    "collusion": ["TOOL_REGISTRY", "TOOL_REPORTS", "TOOL_DRUGS"],
    # Honest: cheap pair to confirm legitimacy
    "none":      ["TOOL_REGISTRY", "TOOL_BILLING"],
}

REASONING_TEMPLATES = {
    "ghost":     "Patient ID returned NO RECORD from the national registry — fabricated patient.",
    "inflation": "Billing system flags claimed amount well above expected cost for severity index.",
    "masking":   "Reports portal shows disclosed comorbidities omit critical conditions on file.",
    "collusion": "Drug DB shows two specialists prescribed the same drug — duplicate prescriber pattern.",
    "none":      "Registry verified, billing within expected range — claim is consistent.",
}


def _expert_actions(deception_type: str) -> list[int]:
    plan = EXPERT_PLAYBOOK.get(deception_type, EXPERT_PLAYBOOK["none"])
    actions = [TOOL_IDX[t] + PanaceaPOMDPEnv.TOOL_ACTION_OFFSET for t in plan]
    verdict_action = (
        PanaceaPOMDPEnv.REJECT if deception_type != "none"
        else PanaceaPOMDPEnv.APPROVE
    )
    actions.append(verdict_action)
    return actions


def rollout_episode(env: PanaceaPOMDPEnv) -> dict:
    obs, info = env.reset()
    deception = info["deception_type"]
    gt = info["ground_truth"]

    initial_prompt = obs["prompt"]
    trace_lines: list[str] = []
    total_reward = 0.0
    tool_cost_total = 0.0

    for action in _expert_actions(deception):
        if action >= PanaceaPOMDPEnv.TOOL_ACTION_OFFSET:
            tool_name = TOOL_NAMES[action - PanaceaPOMDPEnv.TOOL_ACTION_OFFSET]
            trace_lines.append(f"<tool>{tool_name}</tool>")
        else:
            trace_lines.append(f"<think>{REASONING_TEMPLATES[deception]}</think>")

        obs, reward, term, trunc, step_info = env.step(action)
        total_reward += reward

        if step_info.get("status") == "tool_call":
            # Append the evidence line that the tool just produced (last block in context)
            tail = obs["prompt"].split(">> CALL ")[-1]
            evidence = "\n".join(tail.splitlines()[1:])  # skip the ">> CALL X" header line
            trace_lines.append(evidence)
            tool_cost_total += step_info.get("tool_cost", 0.0)
        elif step_info.get("status") == "verdict":
            verdict = step_info["action"]
            trace_lines.append(f"VERDICT: {verdict}")
            trace_lines.append(f"REASONING: {REASONING_TEMPLATES[deception]}")

        if term or trunc:
            break

    return {
        "prompt": initial_prompt,
        "response": "\n".join(trace_lines),
        "ground_truth_label": gt,
        "deception_type": deception,
        "total_reward": round(total_reward, 4),
        "tool_cost_total": round(tool_cost_total, 4),
    }


def harvest(n: int, difficulty: int, out_path: str, seed: int,
            adaptive: bool) -> None:
    env = PanaceaPOMDPEnv(difficulty=difficulty, seed=seed, adaptive=adaptive)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            ep = rollout_episode(env)
            f.write(json.dumps(ep) + "\n")
            written += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n}] last_reward={ep['total_reward']:+.2f} "
                      f"deception={ep['deception_type']}")
    print(f"Wrote {written} trajectories -> {out_path}")
    if env.adaptive_sampler is not None:
        print("Adaptive sampler snapshot:", env.adaptive_sampler.snapshot())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--difficulty", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                   help="Use AdaptiveDeceptionSampler to reweight types (default on)")
    p.add_argument("--out", type=str, default="data/pomdp_trajectories.jsonl")
    args = p.parse_args()
    harvest(args.n, args.difficulty, args.out, args.seed, args.adaptive)


if __name__ == "__main__":
    main()
