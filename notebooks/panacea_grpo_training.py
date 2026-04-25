"""
Panacea GRPO Training Script — Ready for Google Colab
=====================================================

Copy-paste this into Colab cells (split at the # ── CELL markers).
Uses Unsloth + HuggingFace TRL to train a small LLM as an oversight agent
that detects deception in hospital resource claims.

Requirements: Colab with GPU (T4 free tier works with 1.5B model)
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 1: Install Dependencies ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# !pip install unsloth trl openenv-core pydantic matplotlib --quiet
# # Clone your repo (or upload openenv_panacea/ folder)
# !git clone https://github.com/YOUR_USERNAME/panacea.git
# %cd panacea


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 2: Load Model with Unsloth ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # auto-detect
)

# Apply LoRA for efficient training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 3: Generate Training Dataset ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, ".")

from openenv_panacea.scenario_generator import generate_dataset
from openenv_panacea.reward import compute_reward

# Generate 200 adversarial scenarios
scenarios = generate_dataset(n=200, seed=42)

# Format as TRL-compatible dataset
from datasets import Dataset

train_data = Dataset.from_dict({
    "prompt": [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in scenarios
    ],
    "expected_verdict": [s["expected_verdict"] for s in scenarios],
    "deception_type": [s["deception"]["type"] for s in scenarios],
})

print(f"Training dataset: {len(train_data)} scenarios")
print(f"Deception distribution:")
from collections import Counter
dist = Counter(train_data["deception_type"])
for dtype, count in dist.items():
    print(f"  {dtype}: {count} ({count/len(train_data)*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 4: Define Reward Functions ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

import re

def extract_verdict_and_reasoning(text: str) -> tuple[str, str]:
    """Parse VERDICT and REASONING from LLM output."""
    verdict = "REJECTED"  # Default safe
    reasoning = ""

    # Try to find VERDICT: line
    verdict_match = re.search(r"VERDICT:\s*(APPROVED|REJECTED)", text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()

    # Try to find REASONING: line
    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return verdict, reasoning


def oversight_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    GRPO reward function.
    Scores each LLM completion against the ground truth.
    """
    expected_verdicts = kwargs.get("expected_verdict", [])
    deception_types = kwargs.get("deception_type", [])

    rewards = []
    for i, completion in enumerate(completions):
        verdict, reasoning = extract_verdict_and_reasoning(completion)
        expected = expected_verdicts[i] if i < len(expected_verdicts) else "REJECTED"
        d_type = deception_types[i] if i < len(deception_types) else "none"

        reward = compute_reward(
            verdict=verdict,
            expected_verdict=expected,
            deception_type=d_type,
            reasoning=reasoning,
        )
        rewards.append(reward)

    return rewards


def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Bonus reward for following the correct output format."""
    rewards = []
    for completion in completions:
        has_verdict = bool(re.search(r"VERDICT:\s*(APPROVED|REJECTED)", completion, re.IGNORECASE))
        has_reasoning = bool(re.search(r"REASONING:", completion, re.IGNORECASE))
        reward = 0.0
        if has_verdict:
            reward += 0.5
        if has_reasoning:
            reward += 0.3
        rewards.append(reward)
    return rewards


# Quick test
test_completion = "VERDICT: REJECTED\nREASONING: No patient record found, this is a ghost patient."
test_reward = oversight_reward_fn(
    [test_completion], [""],
    expected_verdict=["REJECTED"],
    deception_type=["ghost"],
)
print(f"Test reward: {test_reward[0]} (expected ~2.45)")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 5: Configure GRPO Training ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="./panacea_grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_generations=4,          # Generate 4 completions per prompt
    max_completion_length=256,  # Max tokens for verdict + reasoning
    max_prompt_length=1024,     # Max tokens for the claim prompt
    logging_steps=5,
    save_steps=50,
    seed=42,
    report_to="none",          # Disable wandb
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_data,
    reward_funcs=[oversight_reward_fn, format_reward_fn],
)

print("GRPO Trainer configured. Ready to train!")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Generations per prompt: {training_args.num_generations}")
print(f"  Learning rate: {training_args.learning_rate}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 6: Train! ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

train_result = trainer.train()

print(f"\nTraining complete!")
print(f"  Total steps: {train_result.global_step}")
print(f"  Final loss: {train_result.training_loss:.4f}")

# Save the trained model
trainer.save_model("./panacea_oversight_model")
print("Model saved to ./panacea_oversight_model")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 7: Plot Reward Curves ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import json

# Extract training metrics
log_history = trainer.state.log_history

steps = []
rewards = []
losses = []

for entry in log_history:
    if "loss" in entry:
        steps.append(entry.get("step", 0))
        losses.append(entry["loss"])
    if "reward" in entry or "rewards/oversight_reward_fn" in entry:
        reward_val = entry.get("reward", entry.get("rewards/oversight_reward_fn", 0))
        rewards.append(reward_val)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reward curve
if rewards:
    axes[0].plot(rewards, color="#00d4aa", linewidth=2)
    axes[0].set_title("Oversight Agent Reward Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Average Reward")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

# Loss curve
if losses:
    axes[1].plot(steps[:len(losses)], losses, color="#ff6b6b", linewidth=2)
    axes[1].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

plt.suptitle("Project Panacea — GRPO Training Results", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("Reward curve saved to reward_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 8: Evaluate Before vs After ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  EVALUATION: Before vs After Training")
print("=" * 60)

# Generate test scenarios (unseen during training)
test_scenarios = generate_dataset(n=50, seed=9999)

correct = 0
total_reward = 0.0
results_by_type = {"ghost": [], "inflation": [], "masking": [], "none": []}

for s in test_scenarios:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": s["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
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
    import subprocess

    # Always save to /content/ so it's visible in Colab file browser
    SAVE_DIR = "/content/panacea_oversight_model"
    ZIP_PATH = "/content/panacea_oversight_model.zip"

    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to: {SAVE_DIR}")

    # List what was saved
    result = subprocess.run(["ls", "-lh", SAVE_DIR], capture_output=True, text=True)
    print(result.stdout)

    subprocess.run(["zip", "-r", ZIP_PATH, SAVE_DIR], check=True)
    print(f"Zipped to: {ZIP_PATH}")

    # Verify zip exists and show size
    size = subprocess.run(["du", "-sh", ZIP_PATH], capture_output=True, text=True)
    print(f"Zip size: {size.stdout.strip()}")
    print("\nDownload: Colab left sidebar → folder icon → right-click panacea_oversight_model.zip → Download")

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
