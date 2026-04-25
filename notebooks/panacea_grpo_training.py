"""
Panacea GRPO Training Script — Ready for Google Colab
=====================================================

Copy-paste this into Colab cells (split at the # ── CELL markers).
Uses Unsloth + HuggingFace TRL to train a small LLM as an oversight agent
that detects deception in hospital resource claims.

KEY IMPROVEMENT: Uses the unified ScenarioGenerator which bridges
the GPT-4o specialist reasoning path with the RL training loop.
Generates mixed-difficulty datasets for robust training.

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
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 3: Generate Mixed-Difficulty Training Dataset ────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, ".")

from src.training.scenario_generator import ScenarioGenerator

gen = ScenarioGenerator(seed=42)

# Mixed-difficulty dataset: 300 easy + 150 medium + 50 hard
scenarios = []
scenarios.extend(gen.generate_dataset(n=300, difficulty=1, seed=42))    # Static, fast
scenarios.extend(gen.generate_dataset(n=150, difficulty=2, seed=142))   # Richer templates
scenarios.extend(gen.generate_dataset(n=50,  difficulty=3, seed=242))   # Collusion + multi-specialist

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
    "expected_verdict": [s["ground_truth_label"] for s in scenarios],
    "deception_type": [s["deception_type"] for s in scenarios],
    "difficulty": [s["difficulty"] for s in scenarios],
})

print(f"\nTraining dataset: {len(train_data)} scenarios")
print(f"Difficulty distribution:")
from collections import Counter
diff_dist = Counter(train_data["difficulty"])
for d, count in sorted(diff_dist.items()):
    print(f"  Level {d}: {count} ({count/len(train_data)*100:.0f}%)")
print(f"\nDeception distribution:")
dec_dist = Counter(train_data["deception_type"])
for dtype, count in dec_dist.items():
    print(f"  {dtype}: {count} ({count/len(train_data)*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 4: Define Reward Functions ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

import re

# Zero-sum reward table (from arena.py:_score_match)
REWARD_TABLE = {
    ("REJECTED", "REJECTED"): +2.0,   # Correct catch
    ("APPROVED", "REJECTED"): -3.0,   # Missed fraud (worst case)
    ("APPROVED", "APPROVED"): +1.0,   # Correct approval
    ("REJECTED", "APPROVED"): -2.0,   # False rejection
}


def extract_verdict_and_reasoning(text: str) -> tuple[str, str]:
    """Parse VERDICT and REASONING from LLM output."""
    verdict = "REJECTED"  # Default safe
    reasoning = ""

    verdict_match = re.search(r"VERDICT:\s*(APPROVED|REJECTED)", text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()

    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return verdict, reasoning


def oversight_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Primary GRPO reward function.
    Scores each LLM completion against the ground truth using arena.py reward table.
    """
    expected_verdicts = kwargs.get("expected_verdict", [])
    deception_types = kwargs.get("deception_type", [])

    rewards = []
    for i, completion in enumerate(completions):
        verdict, reasoning = extract_verdict_and_reasoning(completion)
        expected = expected_verdicts[i] if i < len(expected_verdicts) else "REJECTED"
        d_type = deception_types[i] if i < len(deception_types) else "none"

        # Core reward from zero-sum table
        reward = REWARD_TABLE.get((verdict, expected), 0.0)

        # Bonus: correct fraud type identification in reasoning
        if d_type != "none" and verdict == "REJECTED":
            reasoning_lower = reasoning.lower()
            if d_type == "ghost" and ("ghost" in reasoning_lower or "not found" in reasoning_lower):
                reward += 0.5
            elif d_type == "inflation" and ("inflat" in reasoning_lower or "excessive" in reasoning_lower):
                reward += 0.5
            elif d_type == "masking" and ("mask" in reasoning_lower or "hidden" in reasoning_lower or "omit" in reasoning_lower):
                reward += 0.5
            elif d_type == "collusion" and ("collus" in reasoning_lower or "same drug" in reasoning_lower or "identical" in reasoning_lower):
                reward += 0.5

        # Bonus: reasoning depth (>50 tokens)
        if len(reasoning.split()) > 50:
            reward += 0.1

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
test_completion = "VERDICT: REJECTED\nREASONING: No patient record found in the hospital registry. This is a ghost patient fabrication."
test_reward = oversight_reward_fn(
    [test_completion], [""],
    expected_verdict=["REJECTED"],
    deception_type=["ghost"],
)
print(f"\nTest reward: {test_reward[0]} (expected ~2.6)")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 5: Configure GRPO Training ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="./panacea_grpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
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
print(f"  Dataset size: {len(train_data)}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
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

plt.suptitle("Project Panacea — GRPO Training Results (Mixed Difficulty)", fontsize=16, fontweight="bold")
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

# Generate test scenarios at all 3 difficulty levels
test_gen = ScenarioGenerator(seed=9999)
test_scenarios = test_gen.generate_dataset(n=50, difficulty=3, seed=9999)

correct = 0
total_reward = 0.0
results_by_type = {"ghost": [], "inflation": [], "masking": [], "collusion": [], "none": []}

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
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    verdict, reasoning = extract_verdict_and_reasoning(response)

    reward = REWARD_TABLE.get((verdict, s["ground_truth_label"]), 0.0)

    is_correct = (verdict == s["ground_truth_label"])
    correct += int(is_correct)
    total_reward += reward
    dtype = s["deception_type"]
    if dtype in results_by_type:
        results_by_type[dtype].append(is_correct)

print(f"\n  Overall Accuracy: {correct}/{len(test_scenarios)} ({correct/len(test_scenarios)*100:.1f}%)")
print(f"  Average Reward:  {total_reward/len(test_scenarios):+.3f}")
print(f"\n  Accuracy by deception type:")
for dtype, results in results_by_type.items():
    if results:
        acc = sum(results) / len(results) * 100
        print(f"    {dtype:12s}: {sum(results)}/{len(results)} ({acc:.0f}%)")

print(f"\n{'=' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 9: Serve Model via FastAPI (for hackathon demo) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Uncomment these lines in Colab to serve the model for the live demo:

# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn, nest_asyncio, threading
# from pyngrok import ngrok
#
# nest_asyncio.apply()
#
# app = FastAPI(title="Panacea Oversight Model")
#
# class InferenceRequest(BaseModel):
#     prompt: str
#
# @app.post("/generate")
# def generate(req: InferenceRequest):
#     inputs = tokenizer(
#         tokenizer.apply_chat_template(
#             [{"role": "user", "content": req.prompt}],
#             tokenize=False, add_generation_prompt=True,
#         ),
#         return_tensors="pt",
#     ).to(model.device)
#
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
#
#     text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#     return {"text": text}
#
# # Start ngrok tunnel + server
# public_url = ngrok.connect(8000)
# print(f"\n{'='*60}")
# print(f"  MODEL ENDPOINT: {public_url}/generate")
# print(f"  Set this in your .env: OVERSIGHT_ENDPOINT={public_url}/generate")
# print(f"{'='*60}\n")
#
# threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000}).start()
