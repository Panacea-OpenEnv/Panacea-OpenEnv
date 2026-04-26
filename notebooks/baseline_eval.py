"""
Baseline eval — UNTRAINED Qwen2.5-1.5B on the Panacea test split.

Run this in a fresh Colab cell / fresh Python process. It loads the base
model with no LoRA adapter, runs per-deception-type evaluation on the same
test split your trainer used (same seed=42 → identical 150-row test set),
and saves panacea_grpo_out/baseline_eval.json.

Later, after training finishes, the comparison plotter in
panacea_grpo_training.py reads baseline_eval.json + trained_eval.json
and renders the before/after chart.

Usage in Colab:
    %cd /content/Panacea-OpenEnv
    !python notebooks/baseline_eval.py
"""

import os
import re
import json
import random
from collections import Counter

# ── Same prompt + parsers as the trainer (kept self-contained on purpose) ────

SYSTEM_PROMPT = (
    "You are a hospital oversight AI investigating a resource claim under partial "
    "observability. You can call enterprise APIs (TOOL_REGISTRY, TOOL_VITALS, "
    "TOOL_REPORTS, TOOL_DRUGS, TOOL_BILLING) by emitting <tool>NAME</tool>. "
    "Each call has a cost; reliability varies. After enough evidence, finish with:\n"
    "VERDICT: APPROVED or REJECTED\nREASONING: <concise explanation>"
)

_TOOL_TAG_RE = re.compile(r"<tool>\s*(TOOL_[A-Z_]+)\s*</tool>", re.IGNORECASE)

TOOL_COSTS = {
    "TOOL_REGISTRY": -0.15, "TOOL_VITALS": -0.15, "TOOL_REPORTS": -0.25,
    "TOOL_DRUGS": -0.20, "TOOL_BILLING": -0.25,
}
REPEAT_TOOL_PENALTY = -0.05
_PRIMARY_TOOL = {"ghost": "TOOL_REGISTRY", "inflation": "TOOL_BILLING",
                 "masking": "TOOL_REPORTS", "collusion": "TOOL_DRUGS"}
_PRIMARY_FLAG = {"ghost": "NO RECORD", "inflation": "<RATIO=",
                 "masking": "comorbidities_disclosed:", "collusion": "<DUPLICATE-PRESCRIBER>"}
_LEGACY_KEYWORDS = {
    "ghost":     ("ghost", "not found", "no patient", "doesn't exist", "fabricat"),
    "inflation": ("inflat", "overcharg", "excessive", "too high", "above expected"),
    "masking":   ("mask", "hidden", "omit", "missing comorbid", "concealed"),
    "collusion": ("collus", "same drug", "identical", "duplicate"),
}


def parse_verdict(text):
    m = re.search(r"VERDICT:\s*(APPROVED|REJECTED)", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def parse_reasoning(text):
    m = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_tools_called(text):
    return [m.group(1).upper() for m in _TOOL_TAG_RE.finditer(text)]


def _evidence_supports(deception_type, completion):
    if deception_type not in _PRIMARY_TOOL:
        return False
    if _PRIMARY_TOOL[deception_type] not in extract_tools_called(completion):
        return False
    return _PRIMARY_FLAG[deception_type] in completion


def compute_accuracy_reward(verdict, expected_verdict, deception_type, completion):
    if verdict not in ("APPROVED", "REJECTED"):
        return -0.5
    is_deceptive   = (expected_verdict == "REJECTED")
    agent_rejected = (verdict == "REJECTED")
    reward = 0.0
    if is_deceptive:
        if agent_rejected:
            reward += 2.0
            if _evidence_supports(deception_type, completion):
                reward += 0.5
            else:
                reasoning = parse_reasoning(completion).lower()
                kws = _LEGACY_KEYWORDS.get(deception_type, ())
                if kws and any(w in reasoning for w in kws):
                    reward += 0.25
        else:
            reward -= 3.0
    else:
        reward = 1.0 if not agent_rejected else -2.0
    return round(reward, 4)


def replay_tool_costs(completion):
    seen, total = set(), 0.0
    for tool_name in extract_tools_called(completion):
        cost = TOOL_COSTS.get(tool_name, -0.10)
        if tool_name in seen:
            cost += REPEAT_TOOL_PENALTY
        seen.add(tool_name)
        total += cost
    return round(total, 4)


# ── Load BASE model (no LoRA adapter) ────────────────────────────────────────

def load_base_model():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        max_seq_length  = 2048,
        load_in_4bit    = True,
        dtype           = None,
    )
    FastLanguageModel.for_inference(model)
    print(f"Loaded base model: {model.config._name_or_path}  (NO adapter)")
    return model, tokenizer


# ── Reproduce the SAME test split the trainer uses ───────────────────────────

def load_test_split(jsonl_path="data/pomdp_trajectories.jsonl"):
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if "prompt" in ep and "ground_truth_label" in ep:
                episodes.append(ep)
    random.seed(42)              # MUST match the trainer's seed
    random.shuffle(episodes)
    split = max(1, int(len(episodes) * 0.9))
    test_eps = episodes[split:]   # last 10% — same rows the trainer never saw
    print(f"Test split: {len(test_eps)} examples")
    print("Distribution: " +
          ", ".join(f"{k}={v}" for k, v in
                    Counter(e['deception_type'] for e in test_eps).items()))
    return test_eps


def run_inference(model, tokenizer, prompt):
    import torch
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 320,
            temperature    = 0.2,
            top_p          = 0.9,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Eval per deception type ──────────────────────────────────────────────────

def evaluate_per_type(model, tokenizer, test_eps, n_per_type=10):
    by_type, rewards_by_type, tools_by_type = {}, {}, {}

    for ex in test_eps:
        dtype = ex["deception_type"]
        by_type.setdefault(dtype, [])
        rewards_by_type.setdefault(dtype, [])
        tools_by_type.setdefault(dtype, Counter())
        if len(by_type[dtype]) >= n_per_type:
            continue

        response = run_inference(model, tokenizer, ex["prompt"])
        verdict  = parse_verdict(response)
        expected = ex["ground_truth_label"]
        correct  = (verdict is not None) and (verdict == expected)
        by_type[dtype].append(1 if correct else 0)

        acc_r  = compute_accuracy_reward(verdict or "", expected, dtype, response)
        cost_r = replay_tool_costs(response)
        rewards_by_type[dtype].append(acc_r + cost_r)
        tools_by_type[dtype].update(extract_tools_called(response))

        print(f"  [{dtype:10s}] verdict={verdict} expected={expected} "
              f"correct={correct} reward={acc_r + cost_r:+.2f}")

    summary = {
        "tag":                  "baseline",
        "accuracy_per_type":    {k: (sum(v)/len(v) if v else 0.0) for k, v in by_type.items()},
        "mean_reward_per_type": {k: (sum(v)/len(v) if v else 0.0) for k, v in rewards_by_type.items()},
        "n_per_type":           {k: len(v) for k, v in by_type.items()},
        "tool_calls_per_type":  {k: dict(v) for k, v in tools_by_type.items()},
    }

    os.makedirs("./panacea_grpo_out", exist_ok=True)
    out_path = "./panacea_grpo_out/baseline_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("  Baseline (untrained) per-deception-type accuracy")
    print("=" * 50)
    for k, v in summary["accuracy_per_type"].items():
        print(f"  {k:12s}: {v*100:5.1f}%   mean_reward={summary['mean_reward_per_type'][k]:+.2f}")
    overall = sum(summary["accuracy_per_type"].values()) / len(summary["accuracy_per_type"])
    print(f"  {'overall':12s}: {overall*100:5.1f}%")
    print("=" * 50)
    print(f"\nSaved -> {out_path}")
    return summary


if __name__ == "__main__":
    test_eps = load_test_split("data/pomdp_trajectories.jsonl")
    model, tokenizer = load_base_model()
    evaluate_per_type(model, tokenizer, test_eps, n_per_type=10)
