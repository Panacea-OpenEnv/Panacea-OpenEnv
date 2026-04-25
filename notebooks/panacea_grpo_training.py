"""
Panacea GRPO Training — Google Colab
=====================================
Paste each CELL block into a separate Colab cell and run top-to-bottom.
Runtime: GPU (T4 free tier is sufficient for the 1.5B model).
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
# Paste and run this cell first. It takes ~3 minutes on a fresh runtime.

"""
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet
!pip install trl datasets pydantic pyngrok nest_asyncio fastapi uvicorn --quiet
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
    "You are a hospital oversight AI agent. Your job is to review resource "
    "claims from specialist doctors and detect deception: ghost patients, "
    "cost inflation, hidden comorbidities, and collusion.\n\n"
    "Respond with EXACTLY this format and nothing else:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "REASONING: <your concise reasoning>"
)


def extract_verdict_and_reasoning(text: str) -> tuple[str, str]:
    """Parse VERDICT and REASONING from model output."""
    verdict = "REJECTED"  # Safe default
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


# ── Self-contained reward function (no src imports needed) ───────────────────

def compute_reward(
    verdict: str,
    expected_verdict: str,
    deception_type: str,
    reasoning: str = "",
) -> float:
    """
    Score the oversight agent's verdict against ground truth.
    Scoring:
      Correct catch of deception:   +2.0
      Correct approval of clean:    +1.0
      False rejection (blocks legit): -2.0
      Missed deception (approved fraud): -3.0
      +0.5 bonus for naming the deception type in reasoning
    """
    if verdict not in ("APPROVED", "REJECTED"):
        return -0.5

    is_deceptive = (expected_verdict == "REJECTED")
    agent_rejected = (verdict == "REJECTED")
    reward = 0.0

    if is_deceptive:
        if agent_rejected:
            reward += 2.0
            r = reasoning.lower()
            type_keywords = {
                "ghost":     ["ghost", "not found", "no patient", "doesn't exist", "fabricat"],
                "inflation":  ["inflat", "overcharg", "excessive", "too high", "above expected"],
                "masking":    ["mask", "hidden", "omit", "missing comorbid", "concealed"],
                "collusion":  ["collus", "same drug", "identical", "duplicate"],
            }
            if any(w in r for w in type_keywords.get(deception_type, [])):
                reward += 0.5
        else:
            reward -= 3.0
    else:
        reward = 1.0 if not agent_rejected else -2.0

    return round(reward, 4)


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
# CELL 4 — Build training dataset
# Option A: load pre-harvested GPT-4o JSONL (recommended)
# Option B: generate static episodes on-the-fly (fallback, no API needed)
# ══════════════════════════════════════════════════════════════════════════════

# ── Minimal static episode generator (Option B fallback) ─────────────────────

_DECEPTION_TYPES = ["ghost", "inflation", "masking", "collusion", "none", "none"]

_DRUG_POOL = [
    "Aspirin 81mg", "Metformin 500mg", "Lisinopril 10mg", "Amoxicillin 500mg",
    "Atorvastatin 20mg", "Omeprazole 20mg", "Clopidogrel 75mg",
    "Enoxaparin 40mg", "Ceftriaxone 1g", "Metoprolol 50mg",
]
_SPECIALTIES = [
    "Cardiology", "Neurology", "Pulmonology", "Oncology",
    "Orthopedics", "General Medicine", "Nephrology",
]
_BASE_COSTS = {
    "Cardiology": 25000, "Neurology": 30000, "Pulmonology": 20000,
    "Oncology": 50000, "Orthopedics": 18000, "General Medicine": 8000,
    "Nephrology": 22000,
}


def _make_report(spec: str, severity: str = None) -> dict:
    sev = severity or random.choice(["low", "medium", "high", "critical"])
    drug = random.choice(_DRUG_POOL)
    return {
        "specialty": spec,
        "severity": sev,
        "medications": [{"name": drug, "dose": "standard", "frequency": "daily", "duration": "7 days"}],
        "recommended_tests": random.sample(["CBC", "ECG", "MRI", "CT", "X-Ray"], k=2),
        "summary": f"Patient assessed by {spec} specialist. Severity: {sev}. Treatment initiated.",
    }


def _build_prompt(patient_id: str, reports: list[dict], claimed_amount: float) -> str:
    specs = [r["specialty"] for r in reports]
    reports_text = ""
    for i, r in enumerate(reports, 1):
        meds = ", ".join(m["name"] for m in r.get("medications", []))
        reports_text += (
            f"\nSPECIALIST REPORT #{i} — {r['specialty']}:\n"
            f"  Severity: {r['severity']}\n"
            f"  Medications: {meds}\n"
            f"  Assessment: {r.get('summary', '')[:150]}"
        )
    return (
        f"You are a hospital oversight AI agent. Analyze this claim for deception.\n"
        f"\nRESOURCE CLAIM:\n"
        f"  Patient ID: {patient_id}\n"
        f"  Specialists: {specs}\n"
        f"  Claimed Amount: ${claimed_amount:,.2f}"
        f"{reports_text}\n"
        f"\nRespond with:\n"
        f"VERDICT: APPROVED or REJECTED\n"
        f"REASONING: <your detailed reasoning>"
    )


def _generate_episode(deception: str) -> dict:
    spec = random.choice(_SPECIALTIES)
    base = _BASE_COSTS[spec]
    pid = f"P{random.randint(1000, 9999)}"

    if deception == "ghost":
        pid = f"P{random.randint(9000, 9999)}"  # Non-existent patient
        report = _make_report(spec, severity="critical")
        claimed = round(random.uniform(30000, 80000), 2)
        reports = [report]
        label = "REJECTED"

    elif deception == "inflation":
        report = _make_report(spec)
        severity_idx = 0.4
        claimed = round(base * severity_idx * random.uniform(2.5, 4.0), 2)
        report["severity"] = "critical"
        reports = [report]
        label = "REJECTED"

    elif deception == "masking":
        report = _make_report(spec)
        # Summary omits the critical comorbidity
        report["summary"] = f"Patient assessed by {spec}. No significant comorbidities noted."
        claimed = round(base * 0.5 * random.uniform(0.9, 1.1), 2)
        reports = [report]
        label = "REJECTED"

    elif deception == "collusion":
        shared_drug = "Enoxaparin 40mg"
        spec2 = random.choice([s for s in _SPECIALTIES if s != spec])
        r1 = _make_report(spec)
        r2 = _make_report(spec2)
        # Inject same drug into both reports
        r1["medications"].append({"name": shared_drug, "dose": "40mg", "frequency": "daily", "duration": "7 days"})
        r2["medications"].append({"name": shared_drug, "dose": "40mg", "frequency": "daily", "duration": "7 days"})
        claimed = round((base + _BASE_COSTS[spec2]) * 0.6 * random.uniform(1.0, 1.3), 2)
        reports = [r1, r2]
        label = "REJECTED"

    else:  # none — clean claim
        report = _make_report(spec)
        severity_idx = random.uniform(0.2, 0.8)
        claimed = round(base * severity_idx * random.uniform(0.9, 1.1), 2)
        reports = [report]
        label = "APPROVED"

    return {
        "prompt": _build_prompt(pid, reports, claimed),
        "ground_truth_label": label,
        "deception_type": deception,
    }


def build_dataset_from_jsonl(jsonl_path: str):
    """Load pre-harvested GPT-4o episodes from JSONL (Option A)."""
    from datasets import Dataset, DatasetDict
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            ep = json.loads(line.strip())
            if "prompt" in ep and "ground_truth_label" in ep:
                episodes.append(ep)

    random.shuffle(episodes)
    split = int(len(episodes) * 0.9)
    train_eps, test_eps = episodes[:split], episodes[split:]

    def to_dataset(eps):
        return Dataset.from_dict({
            "prompt": [e["prompt"] for e in eps],
            "expected_verdict": [e["ground_truth_label"] for e in eps],
            "deception_type": [e["deception_type"] for e in eps],
        })

    ds = DatasetDict({"train": to_dataset(train_eps), "test": to_dataset(test_eps)})
    print(f"Loaded from JSONL: {len(ds['train'])} train / {len(ds['test'])} test")
    _print_distribution(ds["train"])
    return ds


def build_dataset_static(n: int = 2000, seed: int = 42):
    """Generate static episodes on-the-fly (Option B — no API needed)."""
    from datasets import Dataset, DatasetDict
    random.seed(seed)

    episodes = []
    weights = {"ghost": 0.2, "inflation": 0.2, "masking": 0.2, "collusion": 0.2, "none": 0.2}
    for _ in range(n):
        deception = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        episodes.append(_generate_episode(deception))

    random.shuffle(episodes)
    split = int(n * 0.9)
    train_eps, test_eps = episodes[:split], episodes[split:]

    def to_dataset(eps):
        return Dataset.from_dict({
            "prompt": [e["prompt"] for e in eps],
            "expected_verdict": [e["ground_truth_label"] for e in eps],
            "deception_type": [e["deception_type"] for e in eps],
        })

    ds = DatasetDict({"train": to_dataset(train_eps), "test": to_dataset(test_eps)})
    print(f"Static dataset: {len(ds['train'])} train / {len(ds['test'])} test")
    _print_distribution(ds["train"])
    return ds


def _print_distribution(split):
    dist = Counter(split["deception_type"])
    total = len(split)
    print("Deception distribution:")
    for dtype, count in sorted(dist.items()):
        print(f"  {dtype:12s}: {count:4d}  ({count/total*100:.0f}%)")


# Run ONE of these:
# dataset = build_dataset_from_jsonl("data/gpt4o_reports.jsonl")  # Recommended
dataset = build_dataset_static(n=1000)                          # HACKATHON SPEED: 150 episodes


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Reward functions
# ══════════════════════════════════════════════════════════════════════════════

def make_reward_fns():
    """Return the two GRPO reward functions. All logic is self-contained."""

    def oversight_reward_fn(completions, prompts=None, **kwargs):
        expected_verdicts = kwargs.get("expected_verdict", [])
        deception_types   = kwargs.get("deception_type", [])
        rewards = []
        for i, completion in enumerate(completions):
            verdict, reasoning = extract_verdict_and_reasoning(completion)
            expected  = expected_verdicts[i] if i < len(expected_verdicts) else "REJECTED"
            dec_type  = deception_types[i]   if i < len(deception_types)   else "none"
            rewards.append(compute_reward(verdict, expected, dec_type, reasoning))
        return rewards

    def format_reward_fn(completions, **kwargs):
        rewards = []
        for c in completions:
            has_verdict   = bool(re.search(r"VERDICT:\s*(APPROVED|REJECTED)", c, re.IGNORECASE))
            has_reasoning = bool(re.search(r"REASONING:", c, re.IGNORECASE))
            rewards.append((0.5 if has_verdict else 0.0) + (0.3 if has_reasoning else 0.0))
        return rewards

    # Quick smoke test
    test_out  = "VERDICT: REJECTED\nREASONING: Patient P9999 not found in the registry — ghost patient."
    test_r    = oversight_reward_fn([test_out], expected_verdict=["REJECTED"], deception_type=["ghost"])
    test_fmt  = format_reward_fn([test_out])
    print(f"Reward self-test : {test_r[0]:.2f}  (expected 2.5)")
    print(f"Format self-test : {test_fmt[0]:.2f} (expected 0.8)")

    return oversight_reward_fn, format_reward_fn

# Run:
oversight_reward_fn, format_reward_fn = make_reward_fns()


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — GRPO Training
# ══════════════════════════════════════════════════════════════════════════════

def train(model, tokenizer, dataset):
    from trl import GRPOTrainer, GRPOConfig

    oversight_reward_fn, format_reward_fn = make_reward_fns()

    batch_size = 2
    grad_accum = 2
    total_steps = len(dataset["train"]) // (batch_size * grad_accum)
    save_steps_1_percent = max(1, total_steps // 100)

    args = GRPOConfig(
        output_dir             = "./panacea_grpo_out",
        num_train_epochs       = 1,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate          = 5e-6,
        num_generations        = 2,         # HACKATHON SPEED: Only generate 2 completions per prompt
        max_completion_length  = 64,        # HACKATHON SPEED: Oversight agent only needs ~30 tokens for a verdict
        max_prompt_length      = 1024,
        logging_steps          = 1,
        save_steps             = save_steps_1_percent,
        seed                   = 42,
        report_to              = "none",
        remove_unused_columns  = False,  # Required: dataset has extra columns
    )

    trainer = GRPOTrainer(
        model           = model,
        processing_class = tokenizer,      # trl>=0.9 uses processing_class not tokenizer
        args            = args,
        train_dataset   = dataset["train"],
        reward_funcs    = [oversight_reward_fn, format_reward_fn],
    )

    print("Starting GRPO training...")
    print(f"  Train samples : {len(dataset['train'])}")
    print(f"  Batch size    : {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} accum")
    print(f"  Generations   : {args.num_generations} per prompt")

    result = trainer.train()
    print(f"\nDone! Steps: {result.global_step} | Loss: {result.training_loss:.4f}")

    trainer.save_model("./panacea_oversight_model")
    tokenizer.save_pretrained("./panacea_oversight_model")
    print("Saved to ./panacea_oversight_model")
    return trainer

# Run:
trainer = train(model, tokenizer, dataset)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Inference helper
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
            max_new_tokens    = 256,
            temperature       = 0.1,
            top_p             = 0.9,
            do_sample         = True,
            pad_token_id      = tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, tokenizer, dataset, n_samples: int = 50):
    test_split = dataset["test"]
    n = min(n_samples, len(test_split))
    samples = test_split.select(range(n))

    results      = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "parse_err": 0}
    fn_breakdown = Counter()

    for i in range(n):
        s        = samples[i]
        response = run_inference(model, tokenizer, s["prompt"])
        verdict  = parse_verdict(response)
        actual   = s["expected_verdict"]
        dtype    = s["deception_type"]

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
    print(f"  Missed fraud: {results['fn']}  <- target: 0")
    print(f"  By type    : {dict(fn_breakdown)}")
    print(f"  Parse errs : {results['parse_err']}")
    print(f"{'='*45}\n")
    return results

# Run:
results = evaluate(model, tokenizer, dataset)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Export & serve with ngrok
# ══════════════════════════════════════════════════════════════════════════════

def export_and_serve(model, tokenizer, ngrok_token: str = "", hf_repo: str = ""):
    """
    Export the model and start a FastAPI server accessible via ngrok.
    Copy the PUBLIC URL and paste it into your .env as OVERSIGHT_ENDPOINT=<url>/generate
    """
    import subprocess, threading
    import nest_asyncio, uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    # ── Save ──────────────────────────────────────────────────────────────────
    SAVE_DIR = "/content/panacea_oversight_model"
    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)

    result = subprocess.run(["du", "-sh", SAVE_DIR], capture_output=True, text=True)
    print(f"Model saved: {result.stdout.strip()}")

    if hf_repo:
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        print(f"Pushed to HuggingFace: https://huggingface.co/{hf_repo}")

    # ── FastAPI ───────────────────────────────────────────────────────────────
    nest_asyncio.apply()
    app = FastAPI(title="Panacea Oversight Agent", version="1.0")

    class Req(BaseModel):
        prompt: str
        temperature: float = 0.1

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "panacea-oversight-v1"}

    @app.post("/generate")
    def generate(req: Req):
        """Primary endpoint — compatible with inference_server.py expectations."""
        raw = run_inference(model, tokenizer, req.prompt)
        return {"text": raw}

    @app.post("/analyze")
    def analyze(req: Req):
        """Structured endpoint for direct use."""
        raw     = run_inference(model, tokenizer, req.prompt)
        verdict = parse_verdict(raw)
        if verdict is None:
            raise HTTPException(422, detail=f"Model returned unparseable output: {raw[:200]}")
        return {
            "verdict":    verdict,
            "reasoning":  parse_reasoning(raw),
            "raw_output": raw,
        }

    # ── ngrok tunnel ──────────────────────────────────────────────────────────
    if ngrok_token:
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = ngrok_token
        public_url = ngrok.connect(8000).public_url
        print(f"\n{'='*55}")
        print(f"  PUBLIC URL : {public_url}")
        print(f"  Swagger    : {public_url}/docs")
        print(f"  /generate  : POST {public_url}/generate")
        print(f"\n  Add to .env:")
        print(f"  OVERSIGHT_ENDPOINT={public_url}/generate")
        print(f"{'='*55}\n")
    else:
        print("Running locally: http://localhost:8000  (no ngrok token provided)")

    # Run server in background thread (daemon=True so Colab cell returns)
    def _run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("Server is running. Send requests to the PUBLIC URL above.")
    # NOTE: do NOT call t.join() — that would block the cell forever.

# Run:
export_and_serve(model, tokenizer, ngrok_token="")
