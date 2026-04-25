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
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    verdict, reasoning = extract_verdict_and_reasoning(response)

    reward = compute_reward(
        verdict=verdict,
        expected_verdict=s["expected_verdict"],
        deception_type=s["deception"]["type"],
        reasoning=reasoning,
    )

    is_correct = (verdict == s["expected_verdict"])
    correct += int(is_correct)
    total_reward += reward
    results_by_type[s["deception"]["type"]].append(is_correct)

print(f"\n  Overall Accuracy: {correct}/{len(test_scenarios)} ({correct/len(test_scenarios)*100:.1f}%)")
print(f"  Average Reward:  {total_reward/len(test_scenarios):+.3f}")
print(f"\n  Accuracy by deception type:")
for dtype, results in results_by_type.items():
    if results:
        acc = sum(results) / len(results) * 100
        print(f"    {dtype:12s}: {sum(results)}/{len(results)} ({acc:.0f}%)")

print(f"\n{'=' * 60}")
