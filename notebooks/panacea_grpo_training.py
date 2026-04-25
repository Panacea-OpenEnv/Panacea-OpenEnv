# -*- coding: utf-8 -*-
"""
Panacea — GRPO Oversight Agent Training
========================================
Model  : Qwen2.5-1.5B-Instruct + LoRA (4-bit)
Method : GRPO (no critic — stable on free T4)
Data   : Sampled directly from PanaceaEnv sub-agent generators
         (same claim format, same deception strategies as real inference)

Upload this file to Google Colab and run:
    python panacea_grpo_training.py

Optional env vars:
    HF_REPO          = "your-hf-username/panacea-oversight-v1"
    NGROK_AUTH_TOKEN = "<your ngrok token>"
    SERVE            = "1"   # start FastAPI server after training
    SKIP_TRAIN       = "1"   # skip training, load saved model instead
    N_EPISODES       = "2000"
"""

# ── Section 1 — Install & GPU check ──────────────────────────────────────────
import subprocess, sys

def install():
    pkgs = [
        'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git',
        'trl', 'peft', 'accelerate', 'bitsandbytes',
        'transformers', 'datasets',
        'fastapi', 'uvicorn', 'pyngrok', 'nest-asyncio',
    ]
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet'] + pkgs)

try:
    import unsloth  # already installed (running on Colab after first run)
except ImportError:
    print("Installing dependencies...")
    install()

import os, json, random, re, threading
from collections import Counter

import torch
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Section 2 — Standalone data generators (no backend needed) ───────────────
#
# These mirror sub_agent.py + seed_data.py exactly.
# On Colab there is no running FastAPI backend, so we simulate
# the same patient pool and sub-agent strategies offline.

random.seed(42)

DEPARTMENTS = ["Cardiology", "Pulmonology", "Oncology", "Neurology"]

PROTOCOLS = [
    {"id": 1, "diagnosis": "Routine Checkup",     "base_cost": 200.00,   "required_tier": "LOW"},
    {"id": 2, "diagnosis": "Complex Surgery",      "base_cost": 15000.00, "required_tier": "HIGH"},
    {"id": 3, "diagnosis": "Respiratory Failure",  "base_cost": 25000.00, "required_tier": "HIGH"},
    {"id": 4, "diagnosis": "Cardiac Mapping",      "base_cost": 4500.00,  "required_tier": "MED"},
    {"id": 5, "diagnosis": "ACLS",                 "base_cost": 8000.00,  "required_tier": "HIGH"},
    {"id": 6, "diagnosis": "Oncology Protocol",    "base_cost": 12000.00, "required_tier": "HIGH"},
    {"id": 7, "diagnosis": "Neuro Assessment",     "base_cost": 3500.00,  "required_tier": "MED"},
    {"id": 8, "diagnosis": "Pulmonary Rehab",      "base_cost": 2200.00,  "required_tier": "LOW"},
]

RESOURCES = ["R_ICU", "R_ECMO", "R_NURSE", "R_VENTILATOR", "R_MRI", "ECG_machine"]

PATIENT_POOL = [
    {"patient_id": "P1001", "age": 45, "risk_tier": "LOW",  "severity_index": 1.0,
     "comorbidities": []},
    {"patient_id": "P1002", "age": 72, "risk_tier": "HIGH", "severity_index": 2.5,
     "comorbidities": [
         {"condition": "Diabetes",     "multiplier": 1.2,  "is_critical": False},
         {"condition": "Hypertension", "multiplier": 1.15, "is_critical": False},
     ]},
    {"patient_id": "P1003", "age": 30, "risk_tier": "LOW",  "severity_index": 0.8,
     "comorbidities": [
         {"condition": "Hemophilia", "multiplier": 2.0, "is_critical": True},
     ]},
    {"patient_id": "P1004", "age": 55, "risk_tier": "MED",  "severity_index": 1.5,
     "comorbidities": [
         {"condition": "Asthma", "multiplier": 1.1, "is_critical": False},
     ]},
    # Extra patients for variety
    {"patient_id": "P1005", "age": 61, "risk_tier": "MED",  "severity_index": 1.8,
     "comorbidities": [
         {"condition": "CKD",    "multiplier": 1.3,  "is_critical": False},
         {"condition": "Anemia", "multiplier": 1.05, "is_critical": False},
     ]},
    {"patient_id": "P1006", "age": 38, "risk_tier": "LOW",  "severity_index": 0.9,
     "comorbidities": []},
    {"patient_id": "P1007", "age": 78, "risk_tier": "HIGH", "severity_index": 3.1,
     "comorbidities": [
         {"condition": "Heart failure",   "multiplier": 1.5,  "is_critical": True},
         {"condition": "Atrial fibrillation", "multiplier": 1.2, "is_critical": False},
     ]},
    {"patient_id": "P1008", "age": 50, "risk_tier": "MED",  "severity_index": 1.2,
     "comorbidities": [
         {"condition": "Epilepsy", "multiplier": 1.1, "is_critical": False},
     ]},
]


def _true_amount(patient: dict, protocol: dict) -> float:
    mult = patient["severity_index"]
    for c in patient["comorbidities"]:
        mult *= c["multiplier"]
    return round(protocol["base_cost"] * mult, 2)


def generate_claim(patient: dict, strategy: str) -> dict:
    """
    Mirrors SubAgentGenerator.generate_claim() from src/agents/sub_agent.py.
    Returns the same dict structure the environment's _get_obs() formats.
    """
    protocol = random.choice(PROTOCOLS)
    department = random.choice(DEPARTMENTS)
    resource = random.choice(RESOURCES)

    if strategy == "ghost":
        return {
            "patient_id": f"P{random.randint(9000, 9999)}",  # does not exist in DB
            "protocol_id": protocol["id"],
            "protocol": protocol["diagnosis"],
            "department": department,
            "requested_resource": "R_ICU",
            "reported_comorbidities": [],
            "claimed_amount": 50000.00,
            "is_valid": False,
            "deception_type": "ghost",
        }

    true_amt = _true_amount(patient, protocol)
    comorbidities = patient.get("comorbidities", [])

    if strategy == "honest":
        return {
            "patient_id": patient["patient_id"],
            "protocol_id": protocol["id"],
            "protocol": protocol["diagnosis"],
            "department": department,
            "requested_resource": resource,
            "reported_comorbidities": [c["condition"] for c in comorbidities],
            "claimed_amount": true_amt,
            "is_valid": True,
            "deception_type": "none",
        }

    if strategy == "masking":
        # Hide critical comorbidities so an unsafe procedure gets approved
        reported = [c["condition"] for c in comorbidities if not c.get("is_critical")]
        return {
            "patient_id": patient["patient_id"],
            "protocol_id": protocol["id"],
            "protocol": protocol["diagnosis"],
            "department": department,
            "requested_resource": resource,
            "reported_comorbidities": reported,
            "claimed_amount": true_amt,
            "is_valid": False,
            "deception_type": "masking",
        }

    if strategy == "inflation":
        # Inflate amount 4-9× — mirrors EasySubAgentGenerator
        inflate = random.uniform(4.0, 9.0)
        return {
            "patient_id": patient["patient_id"],
            "protocol_id": protocol["id"],
            "protocol": protocol["diagnosis"],
            "department": department,
            "requested_resource": resource,
            "reported_comorbidities": [c["condition"] for c in comorbidities],
            "claimed_amount": round(true_amt * inflate, 2),
            "is_valid": False,
            "deception_type": "inflation",
        }

    if strategy == "collusion":
        # Same resource double-billed across two departments
        return {
            "patient_id": patient["patient_id"],
            "protocol_id": protocol["id"],
            "protocol": protocol["diagnosis"],
            "department": department,
            "requested_resource": "R_ICU",  # contested resource
            "reported_comorbidities": [c["condition"] for c in comorbidities],
            "claimed_amount": true_amt,
            "is_valid": False,
            "deception_type": "collusion",
        }

    raise ValueError(f"Unknown strategy: {strategy}")


def format_claim_as_prompt(claim: dict) -> str:
    """
    Formats a claim dict into the exact text the oversight agent sees at
    inference time — mirrors PanaceaEnv._get_obs() + the prompt in
    onsite_training.py PanaceaRLWrapper.generate_prompt().
    """
    comorbidities_str = (
        ", ".join(claim["reported_comorbidities"])
        if claim["reported_comorbidities"]
        else "None reported"
    )
    return (
        f"Claim ID: {random.randint(10000, 99999)}\n"
        f"Department: {claim['department']}\n"
        f"Patient ID: {claim['patient_id']}\n"
        f"Protocol: {claim['protocol']}\n"
        f"Resource Requested: {claim['requested_resource']}\n"
        f"Claimed Amount: ${claim['claimed_amount']:,.2f}\n"
        f"Reported Comorbidities: {comorbidities_str}"
    )


SYSTEM_PROMPT = (
    "You are the Panacea Oversight Agent — an AI auditor for hospital resource claims.\n"
    "You receive a structured claim from a specialist department. "
    "Your job is to detect deception: ghost patients, cost inflation, "
    "hidden comorbidities (masking), and collusion between departments.\n\n"
    "Respond in EXACTLY this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "REASONING: <your detailed reasoning, at least 3 sentences>"
)


def build_dataset(n: int = 2000) -> list[dict]:
    """
    Generates n episodes using all four deception strategies + honest claims.
    Split: 35% honest, 20% ghost, 20% inflation, 15% masking, 10% collusion.
    Weights reflect real-world fraud base rates (ghost is rarest, inflation most common).
    """
    strategy_weights = {
        "honest":   0.35,
        "ghost":    0.20,
        "inflation":0.20,
        "masking":  0.15,
        "collusion":0.10,
    }
    rows = []
    for strategy, weight in strategy_weights.items():
        count = int(n * weight)
        for _ in range(count):
            patient = random.choice(PATIENT_POOL)
            claim = generate_claim(patient, strategy)
            prompt_text = format_claim_as_prompt(claim)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text},
                ],
                "expected_verdict": "REJECTED" if not claim["is_valid"] else "APPROVED",
                "deception_type":   claim["deception_type"],
            })
    random.shuffle(rows)
    return rows


def build_hf_dataset():
    from datasets import Dataset

    raw = build_dataset(2000)
    ds  = Dataset.from_list(raw).train_test_split(test_size=0.1, seed=42)

    counts = Counter(r["deception_type"] for r in raw)
    print(f"Train: {len(ds['train'])} | Eval: {len(ds['test'])}")
    print(f"Deception type distribution: {dict(counts)}")
    print(f"Verdict split: APPROVED={sum(1 for r in raw if r['expected_verdict']=='APPROVED')} "
          f"REJECTED={sum(1 for r in raw if r['expected_verdict']=='REJECTED')}")
    return ds


# ── Section 3 — Reward Functions ─────────────────────────────────────────────
#
# Output format: VERDICT: APPROVED or REJECTED\nREASONING: ...
# Mirrors the reward logic in src/environment/reward.py calculate_reward().

def parse_verdict(response: str) -> str | None:
    """Extract APPROVED or REJECTED from model response."""
    match = re.search(r"VERDICT\s*:\s*(APPROVED|REJECTED)", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_reasoning(response: str) -> str:
    """Extract the REASONING text."""
    match = re.search(r"REASONING\s*:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def format_reward(completions, **kwargs) -> list[float]:
    """
    +0.3  response follows VERDICT/REASONING format exactly
    -0.5  no VERDICT line at all
    -0.2  VERDICT present but REASONING missing or empty
    """
    rewards = []
    for completion in completions:
        text     = completion[0]["content"] if isinstance(completion, list) else completion
        verdict  = parse_verdict(text)
        reasoning = parse_reasoning(text)

        if verdict is None:
            rewards.append(-0.5)
        elif not reasoning:
            rewards.append(-0.2)
        else:
            rewards.append(0.3)
    return rewards


def decision_reward(completions, expected_verdict, deception_type, **kwargs) -> list[float]:
    """
    Reward table mirrors src/environment/reward.py calculate_reward():

    Correct REJECT on fraud          +2.0   (True Positive)
    Missed fraud  → false APPROVED   -3.0   (False Negative — ghost_allocation equiv)
    Correct APPROVE on honest        +1.0   (True Negative)
    False flag on honest             -2.0   (False Positive — harms valid patients)

    Bonus: named the right deception_type in REASONING  +0.5
    Bonus: REASONING is ≥ 40 words                      +0.1
    """
    rewards = []
    for completion, ev, dt in zip(completions, expected_verdict, deception_type):
        text      = completion[0]["content"] if isinstance(completion, list) else completion
        verdict   = parse_verdict(text)
        reasoning = parse_reasoning(text)
        reward    = 0.0

        if verdict is None:
            rewards.append(-1.0)
            continue

        is_fraud        = (ev == "REJECTED")
        predicted_fraud = (verdict == "REJECTED")

        if   is_fraud and predicted_fraud:        reward += 2.0
        elif is_fraud and not predicted_fraud:    reward -= 3.0   # worst — missed ghost/fraud
        elif not is_fraud and not predicted_fraud: reward += 1.0
        else:                                     reward -= 2.0

        # Bonus: mentioned the deception type by name in reasoning
        if is_fraud and dt != "none" and dt.lower() in reasoning.lower():
            reward += 0.5

        # Bonus: substantive reasoning
        if len(reasoning.split()) >= 40:
            reward += 0.1

        rewards.append(reward)
    return rewards


print("Reward functions loaded.")


# ── Section 4 — Load Model + LoRA ─────────────────────────────────────────────
def load_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length = 2048,
        load_in_4bit   = True,
        fast_inference = False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
    )

    print(model.print_trainable_parameters())
    return model, tokenizer


# ── Section 5 — GRPO Training ─────────────────────────────────────────────────
def train(model, tokenizer, dataset):
    from trl import GRPOTrainer, GRPOConfig

    args = GRPOConfig(
        output_dir                  = "panacea_oversight_model",
        num_train_epochs            = 3,
        per_device_train_batch_size = 2,        # T4 safe: 4-bit + seq 2048
        gradient_accumulation_steps = 8,        # effective batch = 16
        learning_rate               = 2e-5,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_8bit",
        max_grad_norm               = 0.3,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        save_strategy               = "epoch",
        eval_strategy               = "epoch",
        report_to                   = "none",
        # GRPO-specific
        num_generations             = 4,        # responses per prompt compared within group
        max_new_tokens              = 512,
        temperature                 = 0.8,
        top_p                       = 0.9,
        beta                        = 0.04,     # KL penalty — keep low for 1.5B
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [format_reward, decision_reward],
        args             = args,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["test"],
    )

    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    print(f"\nStarting GRPO training...")
    print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")
    print(f"Effective batch size: {eff_batch}")
    print(f"Epochs: {args.num_train_epochs} | LR: {args.learning_rate}\n")

    trainer.train()
    return trainer


# ── Section 6 — Evaluate ──────────────────────────────────────────────────────
def run_inference(model, tokenizer, messages: list[dict]) -> str:
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = 512,
            temperature    = 0.1,
            top_p          = 0.9,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate(model, tokenizer, dataset):
    from unsloth import FastLanguageModel as FLM
    FLM.for_inference(model)

    samples = dataset["test"].select(range(min(50, len(dataset["test"]))))
    results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "parse_errors": 0}
    type_breakdown = Counter()

    for sample in samples:
        response  = run_inference(model, tokenizer, sample["prompt"])
        verdict   = parse_verdict(response)
        actual    = sample["expected_verdict"]
        dec_type  = sample["deception_type"]

        if verdict is None:
            results["parse_errors"] += 1
            continue

        if   actual == "REJECTED" and verdict == "REJECTED": results["tp"] += 1
        elif actual == "APPROVED" and verdict == "APPROVED": results["tn"] += 1
        elif actual == "APPROVED" and verdict == "REJECTED": results["fp"] += 1
        elif actual == "REJECTED" and verdict == "APPROVED":
            results["fn"] += 1
            type_breakdown[dec_type] += 1  # which fraud types are being missed?

    total     = sum(results.values())
    correct   = results["tp"] + results["tn"]
    precision = results["tp"] / max(results["tp"] + results["fp"], 1)
    recall    = results["tp"] / max(results["tp"] + results["fn"], 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\n{'='*45}")
    print(f" Eval Results  (n={total})")
    print(f"{'='*45}")
    print(f" Accuracy  : {correct/total*100:.1f}%")
    print(f" Precision : {precision*100:.1f}%")
    print(f" Recall    : {recall*100:.1f}%")
    print(f" F1 Score  : {f1*100:.1f}%")
    print(f" False Negatives (missed fraud): {results['fn']}  <- must be 0")
    print(f" Missed by type : {dict(type_breakdown)}")
    print(f" Parse errors   : {results['parse_errors']}")
    print(f"{'='*45}\n")


# ── Section 7 — Export ────────────────────────────────────────────────────────
def export_model(model, tokenizer, hf_repo: str = ""):
    SAVE_DIR = "panacea_oversight_model"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved locally: {SAVE_DIR}/")

    import subprocess
    subprocess.run(["zip", "-r", "panacea_oversight_model.zip", SAVE_DIR], check=True)
    print("Zipped: panacea_oversight_model.zip")

    try:
        from google.colab import files
        files.download("panacea_oversight_model.zip")
    except ImportError:
        print("Not in Colab — download the zip manually from the file browser.")

    if hf_repo:
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        print(f"Pushed to: https://huggingface.co/{hf_repo}")


# ── Section 8 — FastAPI server + ngrok (hackathon demo) ───────────────────────
def serve(model, tokenizer, ngrok_token: str = ""):
    import nest_asyncio, uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel as PBase

    nest_asyncio.apply()

    from unsloth import FastLanguageModel as FLM
    FLM.for_inference(model)

    app = FastAPI(title="Panacea Oversight Agent", version="1.0")

    class ClaimRequest(PBase):
        claim_text: str          # same field name as PanaceaEnv observation
        temperature: float = 0.1

    class OversightResponse(PBase):
        verdict: str             # APPROVED or REJECTED
        reasoning: str
        raw_output: str

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "panacea-oversight-v1"}

    @app.post("/analyze", response_model=OversightResponse)
    def analyze(req: ClaimRequest):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": req.claim_text},
        ]
        raw     = run_inference(model, tokenizer, messages)
        verdict = parse_verdict(raw)
        if verdict is None:
            raise HTTPException(status_code=422,
                                detail=f"Model did not return a valid VERDICT. Output: {raw[:300]}")
        return OversightResponse(
            verdict    = verdict,
            reasoning  = parse_reasoning(raw),
            raw_output = raw,
        )

    # ngrok public tunnel
    if ngrok_token:
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = ngrok_token
        public_url = ngrok.connect(8000).public_url
        print(f"\n{'='*50}")
        print(f"PUBLIC URL : {public_url}")
        print(f"Swagger UI : {public_url}/docs")
        print(f"Analyze    : POST {public_url}/analyze")
        print(f"{'='*50}\n")
    else:
        print("Local server: http://localhost:8000  (Swagger: /docs)")

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("Server started.")
    t.join()   # blocks when run as a script; Ctrl-C to stop


# ── Quick demo: show a sample prompt/response before training ─────────────────
def demo_sample(dataset):
    sample = dataset["train"][0]
    print("\n--- Sample training prompt ---")
    for msg in sample["prompt"]:
        print(f"[{msg['role'].upper()}]\n{msg['content']}\n")
    print(f"Expected verdict : {sample['expected_verdict']}")
    print(f"Deception type   : {sample['deception_type']}")
    print("------------------------------\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SKIP_TRAIN  = os.getenv("SKIP_TRAIN",       "0") == "1"
    DO_SERVE    = os.getenv("SERVE",            "0") == "1"
    HF_REPO     = os.getenv("HF_REPO",           "")
    NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN",  "")
    N_EPISODES  = int(os.getenv("N_EPISODES",  "2000"))

    # Build dataset from Panacea sub-agent generators
    dataset = build_hf_dataset()
    demo_sample(dataset)

    if not SKIP_TRAIN:
        model, tokenizer = load_model()
        train(model, tokenizer, dataset)
        export_model(model, tokenizer, hf_repo=HF_REPO)
    else:
        # Load an already-saved checkpoint
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = "panacea_oversight_model",
            max_seq_length = 2048,
            load_in_4bit   = True,
        )

    evaluate(model, tokenizer, dataset)

    if DO_SERVE:
        serve(model, tokenizer, ngrok_token=NGROK_TOKEN)
