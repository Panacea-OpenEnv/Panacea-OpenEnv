"""
Panacea GRPO Training — Google Colab
=====================================
Multi-step POMDP oversight agent training.

Each training example is a tool-call trajectory produced by
src/training/trajectory_harvester.py:

    <tool>TOOL_REGISTRY</tool>
    [REGISTRY] Lookup pid=P9999: NO RECORD FOUND.
    <think>Patient ID does not exist — fabricated.</think>
    VERDICT: REJECTED
    REASONING: Patient ID returned NO RECORD from the national registry.

The trainer rewards the FULL trajectory:
    total_reward = accuracy_reward + sum(tool_costs) + evidence_bonus
matching PanaceaPOMDPEnv.step() exactly — env↔training reward alignment.

Paste each CELL block into a separate Colab cell and run top-to-bottom.
Runtime: GPU (T4 free tier sufficient for the 1.5B model).
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
# Paste and run this cell first. ~3 minutes on a fresh runtime.

"""
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
!pip install trl datasets pydantic pyngrok nest_asyncio fastapi uvicorn matplotlib --quiet
"""


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Shared utilities (no external imports)
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import json
import random
from collections import Counter

SYSTEM_PROMPT = (
    "You are a hospital oversight AI investigating a resource claim under partial "
    "observability. You can call enterprise APIs (TOOL_REGISTRY, TOOL_VITALS, "
    "TOOL_REPORTS, TOOL_DRUGS, TOOL_BILLING) by emitting <tool>NAME</tool>. "
    "Each call has a cost; reliability varies. After enough evidence, finish with:\n"
    "VERDICT: APPROVED or REJECTED\nREASONING: <concise explanation>"
)

# Tool-trace tag we expect in completions
_TOOL_TAG_RE = re.compile(r"<tool>\s*(TOOL_[A-Z_]+)\s*</tool>", re.IGNORECASE)


def extract_verdict_and_reasoning(text: str) -> tuple[str, str]:
    """Parse VERDICT and REASONING. Returns ('', '') if no verdict tag —
    the reward function treats that as a malformed completion (penalty)."""
    verdict = ""
    reasoning = ""
    m = re.search(r"VERDICT:\s*(APPROVED|REJECTED)", text, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
    m = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
    return verdict, reasoning


def parse_verdict(text: str) -> str | None:
    m = re.search(r"VERDICT:\s*(APPROVED|REJECTED)", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def parse_reasoning(text: str) -> str:
    m = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_tools_called(text: str) -> list[str]:
    """Return the ordered list of tool names invoked in the trajectory."""
    return [m.group(1).upper() for m in _TOOL_TAG_RE.finditer(text)]


# ── Tool-call cost table (matches src/environment/tool_backends.py) ──────────

TOOL_COSTS: dict[str, float] = {
    "TOOL_REGISTRY": -0.15,   # base -0.10 + latency -0.05
    "TOOL_VITALS":   -0.15,
    "TOOL_REPORTS":  -0.25,   # base -0.15 + latency -0.10
    "TOOL_DRUGS":    -0.20,
    "TOOL_BILLING":  -0.25,
}
REPEAT_TOOL_PENALTY = -0.05


# ── Evidence-grounded reward (mirrors src/environment/reward.py) ─────────────

_PRIMARY_TOOL = {
    "ghost": "TOOL_REGISTRY", "inflation": "TOOL_BILLING",
    "masking": "TOOL_REPORTS", "collusion": "TOOL_DRUGS",
}
_PRIMARY_FLAG = {
    "ghost": "NO RECORD", "inflation": "<RATIO=",
    "masking": "comorbidities_disclosed:", "collusion": "<DUPLICATE-PRESCRIBER>",
}
_LEGACY_KEYWORDS = {
    "ghost":     ("ghost", "not found", "no patient", "doesn't exist", "fabricat"),
    "inflation": ("inflat", "overcharg", "excessive", "too high", "above expected"),
    "masking":   ("mask", "hidden", "omit", "missing comorbid", "concealed"),
    "collusion": ("collus", "same drug", "identical", "duplicate"),
}


def _evidence_supports(deception_type: str, completion: str) -> bool:
    """True iff the canonical tool was called AND the completion contains the
    canonical evidence flag from that tool."""
    if deception_type not in _PRIMARY_TOOL:
        return False
    needed_tool = _PRIMARY_TOOL[deception_type]
    needed_flag = _PRIMARY_FLAG[deception_type]
    if needed_tool not in extract_tools_called(completion):
        return False
    return needed_flag in completion


def compute_accuracy_reward(verdict: str, expected_verdict: str,
                            deception_type: str, completion: str) -> float:
    """Per-trajectory accuracy + evidence-grounded bonus.

    No step penalty here — the tool-cost replay term already accounts for
    trajectory length. Total reward = this + sum(tool_costs).
    """
    if verdict not in ("APPROVED", "REJECTED"):
        return -0.5

    is_deceptive = (expected_verdict == "REJECTED")
    agent_rejected = (verdict == "REJECTED")
    reward = 0.0

    if is_deceptive:
        if agent_rejected:
            reward += 2.0
            if _evidence_supports(deception_type, completion):
                reward += 0.5  # full evidence-grounded bonus
            else:
                # half-credit fallback for legacy keyword match
                reasoning = parse_reasoning(completion).lower()
                kws = _LEGACY_KEYWORDS.get(deception_type, ())
                if kws and any(w in reasoning for w in kws):
                    reward += 0.25
        else:
            reward -= 3.0
    else:
        reward = 1.0 if not agent_rejected else -2.0

    return round(reward, 4)


def replay_tool_costs(completion: str) -> float:
    """Sum up the cost of every <tool>X</tool> call in the completion.
    Repeat calls incur the repeat penalty just like the env."""
    seen: set[str] = set()
    total = 0.0
    for tool_name in extract_tools_called(completion):
        cost = TOOL_COSTS.get(tool_name, -0.10)
        if tool_name in seen:
            cost += REPEAT_TOOL_PENALTY
        seen.add(tool_name)
        total += cost
    return round(total, 4)


print("Cell 2 ready.")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Load model with Unsloth
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.config._name_or_path}")
    print(f"Trainable parameters: {trainable:,}")
    return model, tokenizer

# Run:
model, tokenizer = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Load POMDP trajectory dataset
# ══════════════════════════════════════════════════════════════════════════════
# Expects data/pomdp_trajectories.jsonl produced by:
#   python -m src.training.trajectory_harvester --n 1500 --difficulty 3
# Each line: {prompt, response, ground_truth_label, deception_type, total_reward, tool_cost_total}

def build_dataset_from_jsonl(jsonl_path: str = "data/pomdp_trajectories.jsonl"):
    from datasets import Dataset, DatasetDict
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if "prompt" in ep and "ground_truth_label" in ep:
                episodes.append(ep)

    random.seed(42)
    random.shuffle(episodes)
    split = max(1, int(len(episodes) * 0.9))
    train_eps, test_eps = episodes[:split], episodes[split:]

    def to_dataset(eps):
        return Dataset.from_dict({
            "prompt":           [e["prompt"] for e in eps],
            "response":         [e.get("response", "") for e in eps],
            "expected_verdict": [e["ground_truth_label"] for e in eps],
            "deception_type":   [e["deception_type"] for e in eps],
            "expert_total_reward": [e.get("total_reward", 0.0) for e in eps],
        })

    ds = DatasetDict({"train": to_dataset(train_eps), "test": to_dataset(test_eps)})
    print(f"Loaded JSONL: {len(ds['train'])} train / {len(ds['test'])} test")
    _print_distribution(ds["train"])
    return ds


def _print_distribution(split):
    dist = Counter(split["deception_type"])
    total = len(split)
    print("Deception distribution:")
    for dtype, count in sorted(dist.items()):
        print(f"  {dtype:12s}: {count:4d}  ({count/total*100:.0f}%)")


# Run:
dataset = build_dataset_from_jsonl("data/pomdp_trajectories.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Reward functions
# ══════════════════════════════════════════════════════════════════════════════
# Three reward functions (composed by GRPO):
#   1. tool_trace_reward_fn — accuracy + evidence + replayed tool costs
#                             (matches PanaceaPOMDPEnv.step exactly)
#   2. format_reward_fn     — small bonus for valid VERDICT/REASONING tags
#   3. tool_use_reward_fn   — small bonus for calling at least one tool
#                             (encourages investigation over guessing)

def make_reward_fns():

    def tool_trace_reward_fn(completions, prompts=None, **kwargs):
        expected_verdicts = kwargs.get("expected_verdict", [])
        deception_types   = kwargs.get("deception_type", [])
        rewards = []
        for i, completion in enumerate(completions):
            verdict, _ = extract_verdict_and_reasoning(completion)
            expected = expected_verdicts[i] if i < len(expected_verdicts) else "REJECTED"
            dtype    = deception_types[i]   if i < len(deception_types)   else "none"
            acc = compute_accuracy_reward(verdict, expected, dtype, completion)
            tool_cost_sum = replay_tool_costs(completion)
            rewards.append(round(acc + tool_cost_sum, 4))
        return rewards

    def format_reward_fn(completions, **kwargs):
        rewards = []
        for c in completions:
            has_verdict   = bool(re.search(r"VERDICT:\s*(APPROVED|REJECTED)", c, re.IGNORECASE))
            has_reasoning = bool(re.search(r"REASONING:", c, re.IGNORECASE))
            rewards.append((0.3 if has_verdict else 0.0) + (0.2 if has_reasoning else 0.0))
        return rewards

    def tool_use_reward_fn(completions, **kwargs):
        # +0.2 for calling at least one tool, encourages investigation behavior
        return [0.2 if extract_tools_called(c) else -0.2 for c in completions]

    # Smoke test
    test_out = (
        "<tool>TOOL_REGISTRY</tool>\n"
        "[REGISTRY] Lookup pid=P9999: NO RECORD FOUND.\n"
        "<think>Patient does not exist — ghost.</think>\n"
        "VERDICT: REJECTED\n"
        "REASONING: Registry returned NO RECORD — fabricated patient."
    )
    r1 = tool_trace_reward_fn([test_out], expected_verdict=["REJECTED"], deception_type=["ghost"])
    r2 = format_reward_fn([test_out])
    r3 = tool_use_reward_fn([test_out])
    print(f"tool_trace_reward : {r1[0]:.2f}  (expected ≈ 2.35; 2.0 + 0.5 - 0.15)")
    print(f"format_reward     : {r2[0]:.2f}  (expected 0.5)")
    print(f"tool_use_reward   : {r3[0]:.2f}  (expected 0.2)")

    return tool_trace_reward_fn, format_reward_fn, tool_use_reward_fn

# Run:
tool_trace_reward_fn, format_reward_fn, tool_use_reward_fn = make_reward_fns()


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — SFT warm-up (teach the <tool>X</tool> format)
# ══════════════════════════════════════════════════════════════════════════════
# 50 SFT steps before GRPO so the model can already emit a structured trajectory.
# Without this warm-up, GRPO often spends 200+ steps just discovering the format.

def sft_warmup(model, tokenizer, dataset, steps: int = 50):
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Build conversation-style examples
    rows = []
    for ex in dataset["train"]:
        if not ex["response"]:
            continue
        rows.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": ex["prompt"]},
                {"role": "assistant", "content": ex["response"]},
            ]
        })
    sft_ds = Dataset.from_list(rows)

    args = SFTConfig(
        output_dir                  = "./panacea_sft_warmup",
        num_train_epochs            = 1,
        max_steps                   = steps,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        learning_rate               = 2e-5,
        logging_steps               = 5,
        save_strategy               = "no",
        seed                        = 42,
        report_to                   = "none",
        max_seq_length              = 1536,
    )
    def _formatting_func(examples):
        msgs = examples["messages"]
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
            return [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)]
        return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = args,
        train_dataset    = sft_ds,
        formatting_func  = _formatting_func,
    )
    print(f"SFT warm-up: {steps} steps on {len(sft_ds)} samples...")
    trainer.train()
    print("SFT warm-up complete.")
    return model

# Run:
model = sft_warmup(model, tokenizer, dataset, steps=200)


def sft_sanity_check(model, tokenizer, dataset, n: int = 3):
    """Generate a few samples post-SFT. If completions don't contain VERDICT,
    the warm-up failed and GRPO will not learn — abort early."""
    print("\n=== SFT sanity check ===")
    sample = dataset["train"].select(range(n))
    ok = 0
    for i in range(n):
        out = run_inference(model, tokenizer, sample[i]["prompt"])
        has_verdict = bool(re.search(r"VERDICT:\s*(APPROVED|REJECTED)", out, re.IGNORECASE))
        has_tool    = bool(_TOOL_TAG_RE.search(out))
        print(f"  sample {i}: verdict={has_verdict} tool={has_tool} | preview: {out[:120]!r}")
        if has_verdict: ok += 1
    print(f"=== {ok}/{n} samples emit VERDICT — proceed if ok>=2, otherwise re-run SFT ===\n")

sft_sanity_check(model, tokenizer, dataset)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — GRPO training on POMDP trajectories
# ══════════════════════════════════════════════════════════════════════════════

def train(model, tokenizer, dataset):
    from trl import GRPOTrainer, GRPOConfig

    tool_trace_reward_fn, format_reward_fn, tool_use_reward_fn = make_reward_fns()

    batch_size = 2
    grad_accum = 2
    num_epochs = 1  # 1350 train * 2 / (2*2) = ~1350 steps (>= 1000 target)

    args = GRPOConfig(
        output_dir                  = "./panacea_grpo_out",
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = 5e-6,
        num_generations             = 4,      # need >=2 with variance; 4 is the GRPO sweet spot
        temperature                 = 0.9,    # without sampling temp, all generations are identical → zero advantage
        max_completion_length       = 320,    # expert trajectories avg ~250; 320 leaves headroom for EOS
        max_prompt_length           = 1024,
        logging_steps               = 1,
        save_steps                  = 200,
        seed                        = 42,
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = args,
        train_dataset    = dataset["train"],
        reward_funcs     = [tool_trace_reward_fn, format_reward_fn, tool_use_reward_fn],
    )

    print("Starting GRPO training on POMDP trajectories...")
    print(f"  Train samples : {len(dataset['train'])}")
    print(f"  Epochs        : {num_epochs}")
    print(f"  Batch size    : {batch_size} x {grad_accum} accum")
    print(f"  Generations   : {args.num_generations} per prompt")
    print(f"  Max completion: {args.max_completion_length} tokens")

    result = trainer.train()
    print(f"\nDone! Steps: {result.global_step} | Loss: {result.training_loss:.4f}")

    trainer.save_model("./panacea_oversight_model")
    tokenizer.save_pretrained("./panacea_oversight_model")
    print("Saved to ./panacea_oversight_model")

    # Persist full step-by-step metrics for the report
    save_training_metrics(trainer, out_dir="./panacea_grpo_out")
    return trainer


def save_training_metrics(trainer, out_dir: str = "./panacea_grpo_out"):
    """Dump every logged step (loss + per-reward-fn means) to JSONL/CSV and
    render a loss/reward curve. The `trainer.state.log_history` is the
    canonical source — every `logging_steps` tick appends a dict here."""
    import csv

    os.makedirs(out_dir, exist_ok=True)
    history = trainer.state.log_history or []

    jsonl_path = os.path.join(out_dir, "training_metrics.jsonl")
    with open(jsonl_path, "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    keys = sorted({k for row in history for k in row.keys()})
    csv_path = os.path.join(out_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, "") for k in keys})

    print(f"Wrote {len(history)} log rows -> {jsonl_path}")
    print(f"Columns: {keys}")

    try:
        import matplotlib.pyplot as plt
        loss_steps = [r["step"] for r in history if "loss" in r and "step" in r]
        loss_vals  = [r["loss"] for r in history if "loss" in r and "step" in r]
        reward_keys = [k for k in keys if k.startswith("rewards/") or k == "reward"]

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        if loss_vals:
            ax[0].plot(loss_steps, loss_vals, color="tab:red")
            ax[0].set_ylabel("loss"); ax[0].grid(alpha=0.3)
        for rk in reward_keys:
            xs = [r["step"] for r in history if rk in r and "step" in r]
            ys = [r[rk]     for r in history if rk in r and "step" in r]
            if ys:
                ax[1].plot(xs, ys, label=rk)
        ax[1].set_xlabel("logging step"); ax[1].set_ylabel("reward")
        ax[1].legend(loc="best", fontsize=8); ax[1].grid(alpha=0.3)
        fig.suptitle("GRPO training: loss + per-reward-fn means")
        fig.tight_layout()
        chart = os.path.join(out_dir, "training_curves.png")
        fig.savefig(chart, dpi=120)
        print(f"Saved chart -> {chart}")
    except Exception as e:
        print(f"(plot skipped: {e})")

# Run:
trainer = train(model, tokenizer, dataset)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Inference helper
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, tokenizer, prompt: str) -> str:
    """Run the trained model on a single prompt string."""
    import torch
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 384,
            temperature    = 0.2,
            top_p          = 0.9,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Evaluate + curriculum drift chart
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, tokenizer, dataset, n_samples: int = 50):
    test_split = dataset["test"]
    n = min(n_samples, len(test_split))
    samples = test_split.select(range(n))

    results      = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "parse_err": 0}
    fn_breakdown = Counter()
    tools_per_type: dict[str, Counter] = {}

    for i in range(n):
        s        = samples[i]
        response = run_inference(model, tokenizer, s["prompt"])
        verdict  = parse_verdict(response)
        actual   = s["expected_verdict"]
        dtype    = s["deception_type"]

        # Track which tools the model called per deception type
        tools_per_type.setdefault(dtype, Counter()).update(extract_tools_called(response))

        if verdict is None:
            results["parse_err"] += 1
            continue
        if   actual == "REJECTED" and verdict == "REJECTED": results["tp"] += 1
        elif actual == "APPROVED" and verdict == "APPROVED": results["tn"] += 1
        elif actual == "APPROVED" and verdict == "REJECTED": results["fp"] += 1
        elif actual == "REJECTED" and verdict == "APPROVED":
            results["fn"] += 1
            fn_breakdown[dtype] += 1

    total   = sum(results.values())
    correct = results["tp"] + results["tn"]
    prec    = results["tp"] / max(results["tp"] + results["fp"], 1)
    rec     = results["tp"] / max(results["tp"] + results["fn"], 1)
    f1      = 2 * prec * rec / max(prec + rec, 1e-9)

    print(f"\n{'='*45}")
    print(f"  Eval Results  (n={total})")
    print(f"{'='*45}")
    print(f"  Accuracy   : {correct/total*100:.1f}%")
    print(f"  Precision  : {prec*100:.1f}%")
    print(f"  Recall     : {rec*100:.1f}%")
    print(f"  F1         : {f1*100:.1f}%")
    print(f"  Missed     : {results['fn']}  (target: 0)")
    print(f"  By type    : {dict(fn_breakdown)}")
    print(f"  Parse errs : {results['parse_err']}")
    print(f"{'='*45}\n")

    print("Tools called per deception type:")
    for dtype, cnt in sorted(tools_per_type.items()):
        top = ", ".join(f"{t}={c}" for t, c in cnt.most_common(3))
        print(f"  {dtype:12s}: {top}")
    return results, tools_per_type


def plot_curriculum_drift(log_path: str = "data/curriculum_log.jsonl"):
    """Render adaptive sampler weights over training episodes.
    Run AFTER trajectory harvest (which writes the log) for a non-trivial chart."""
    import matplotlib.pyplot as plt

    if not os.path.exists(log_path):
        print(f"No curriculum log at {log_path}. Run the trajectory harvester first.")
        return

    rows = [json.loads(line) for line in open(log_path) if line.strip()]
    if not rows:
        print("Empty curriculum log.")
        return

    episodes = [r["episode"] for r in rows]
    types = list(rows[0]["weights"].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in types:
        ax.plot(episodes, [r["weights"][t] for r in rows], marker="o", label=t)
    ax.set_xlabel("episode")
    ax.set_ylabel("sampling weight")
    ax.set_title("Adaptive curriculum drift — deception types reweight as agent learns")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("curriculum_drift.png", dpi=120)
    plt.show()
    print("Saved chart -> curriculum_drift.png")


# Run:
results, tools_per_type = evaluate(model, tokenizer, dataset)
plot_curriculum_drift()


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Export & serve with ngrok
# ══════════════════════════════════════════════════════════════════════════════

def export_and_serve(model, tokenizer, ngrok_token: str = "", hf_repo: str = ""):
    """
    Export model and start a FastAPI server accessible via ngrok.
    Copy the PUBLIC URL to .env as OVERSIGHT_ENDPOINT=<url>/generate
    """
    import subprocess, threading
    import nest_asyncio, uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    SAVE_DIR = "/content/panacea_oversight_model"
    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)

    result = subprocess.run(["du", "-sh", SAVE_DIR], capture_output=True, text=True)
    print(f"Model saved: {result.stdout.strip()}")

    if hf_repo:
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        print(f"Pushed to HuggingFace: https://huggingface.co/{hf_repo}")

    nest_asyncio.apply()
    app = FastAPI(title="Panacea Oversight Agent", version="2.0")

    class Req(BaseModel):
        prompt: str
        temperature: float = 0.2

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "panacea-oversight-pomdp-v2"}

    @app.post("/generate")
    def generate(req: Req):
        return {"text": run_inference(model, tokenizer, req.prompt)}

    @app.post("/analyze")
    def analyze(req: Req):
        raw = run_inference(model, tokenizer, req.prompt)
        verdict = parse_verdict(raw)
        if verdict is None:
            raise HTTPException(422, detail=f"Unparseable: {raw[:200]}")
        return {
            "verdict":      verdict,
            "reasoning":    parse_reasoning(raw),
            "tools_called": extract_tools_called(raw),
            "raw_output":   raw,
        }

    if ngrok_token:
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = ngrok_token
        public_url = ngrok.connect(8000).public_url
        print(f"\n{'='*55}")
        print(f"  PUBLIC URL : {public_url}")
        print(f"  /generate  : POST {public_url}/generate")
        print(f"  Add to .env:")
        print(f"  OVERSIGHT_ENDPOINT={public_url}/generate")
        print(f"{'='*55}\n")
    else:
        print("Running locally: http://localhost:8000  (no ngrok token provided)")

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("Server is running.")

# Run:
export_and_serve(model, tokenizer, ngrok_token="")
