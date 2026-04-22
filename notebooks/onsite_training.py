# %% [markdown]
# # Project Panacea: RL Training Pipeline (OpenEnv + Unsloth + TRL)
# 
# **Objective:** Train an Oversight LLM Agent natively inside the Panacea environment to mathematically verify sub-agent requests, catching obfuscated comorbidities and schema drift.
# 
# **Stack:** `Unsloth` (4-bit QLoRA) + `TRL` (PPOTrainer) + `OpenEnv` (Gymnasium)
# 
# ---

# %% [markdown]
# ### Step 1: Environment Setup & Installs
# *Run this cell if executing in Google Colab for the Hackathon Onsite.*

# %%
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
# !pip install gymnasium requests matplotlib rich

# import os
# import sys
# sys.path.append(os.path.abspath('..')) # Assuming project root is accessible

# %% [markdown]
# ### Step 2: Model & Tokenizer Initialization (Unsloth)
# We use `unsloth` to load a 4-bit quantized model (e.g., Llama-3-8B) to fit inside a colab T4 GPU. We then apply LoRA adapters to make it trainable via RL.

# %%
import torch
from unsloth import FastLanguageModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, set_seed
from transformers import AutoTokenizer

print("Loading Unsloth Base Model (4-bit)...")

max_seq_length = 2048 # Adjust based on DB schema sizes
model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# 1. Load the base causal LM
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters for efficient fine-tuning
base_model = FastLanguageModel.get_peft_model(
    base_model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# 3. Wrap with TRL's ValueHead for PPO
# Note: PPOTrainer requires a Value Head to predict advantage (V-function).
rl_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
rl_model.is_peft_model = True # Important for TRL integration

# Ensure pad token exists
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model wrapped with RL Value Head and PEFT adapters enabled.")

# %% [markdown]
# ### Step 3: OpenEnv Integration Wrapper
# The RL loop must capture trajectories (State, Action, Reward). We wrap the `PanaceaEnv` to format text observations into tokenized states for the LLM.

# %%
import sys
import os
# Adjust path assuming this is run from `notebooks/` folder inside Panacea root
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.environment.env import PanaceaEnv

class PanaceaRLWrapper:
    """
    Interfaces between the OpenEnv standard (PanaceaEnv) and TRL Data generation.
    """
    def __init__(self, env, tokenizer):
        self.env = env
        self.tokenizer = tokenizer
        self.system_prompt = "You are the Panacea Oversight Agent. Analyze the hospital claim and determine if it should be APPROVED (1) or REJECTED (0). You must output your thinking, followed by the integer verdict: e.g., <verdict>0</verdict>"
        
    def generate_prompt(self, obs):
        """Formats the observation + trust scores into the Chat Template"""
        claim = obs.get("claim_text", "")
        # Real prompt would include DB context schema here
        prompt = f"{self.system_prompt}\n\n[INCOMING CLAIM]\n{claim}\n\n[YOUR ANALYSIS]:"
        return prompt

    def decode_action(self, response_text):
        """Parses the LLM's text output back into the discrete Action Space [0, 1]"""
        try:
            if "<verdict>1</verdict>" in response_text:
                return 1
            elif "<verdict>0</verdict>" in response_text:
                return 0
            else:
                return -1 # Invalid format penalty
        except Exception:
            return -1

env_wrapper = PanaceaRLWrapper(PanaceaEnv(), tokenizer)

# %% [markdown]
# ### Step 4: The PPO Training Loop (TRL)
# We collect experience from the environment and update the policy to maximize our Panacea reward schema (handling schema drifts and catching omisssions!).

# %%
import matplotlib.pyplot as plt
import numpy as np

# PPO Configuration
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    target_kl=0.1,
    init_kl_coef=0.2,
    seed=42,
)

# Initialize TRL's PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=rl_model,
    ref_model=None, # None uses the same PEFT model with adapters disabled as reference
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 128,
}

num_episodes = 20 # Keep small for hackathon demo
reward_history = []

print("Starting RL Training Loop...")

# Optional: Disable fast inference for training
FastLanguageModel.for_training(rl_model.pretrained_model)

for episode in range(num_episodes):
    obs, info = env_wrapper.env.reset()
    done = False
    
    episode_reward = 0
    query_tensors = []
    response_tensors = []
    rewards = []
    
    # 1. Format State
    prompt_text = env_wrapper.generate_prompt(obs)
    q_tensor = tokenizer.encode(prompt_text, return_tensors="pt").to(ppo_trainer.accelerator.device)[0]
    
    # 2. Generate Action (Trajectory Collection)
    with torch.no_grad():
        r_tensor = ppo_trainer.generate(q_tensor.unsqueeze(0), **generation_kwargs)[0]
    
    response_text = tokenizer.decode(r_tensor, skip_special_tokens=True)
    
    # 3. Environment Step
    action = env_wrapper.decode_action(response_text)
    
    # Send action back to Panacea OpenEnv
    next_obs, reward, done, truncated, step_info = env_wrapper.env.step(action)
    
    # Map back format penalties if needed
    if action == -1:
        reward -= 2.0 # Formatting penalty
        
    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(ppo_trainer.accelerator.device)
    
    # 4. PPO Update Step
    stats = ppo_trainer.step([q_tensor], [r_tensor], [reward_tensor])
    
    reward_history.append(reward)
    
    print(f"Episode {episode+1}/{num_episodes} | Action: {action} | Step Reward: {reward:.2f} | Loss: {stats['ppo/loss/policy']:.4f}")

# Plotting the Learning Curve for Hackathon Judges
plt.figure(figsize=(10, 5))
plt.plot(reward_history, marker='o', linestyle='-', color='b')
plt.title("Oversight Agent PPO Reward Curve (Panacea Environment)")
plt.xlabel("Episode")
plt.ylabel("Reward (Score)")
plt.grid(True)
plt.savefig("rl_learning_curve.png")
print("Saved learning curve to rl_learning_curve.png")

# %% [markdown]
# ### Step 5: Inference & Demo 
# Run a final forward pass with our optimized LoRA weights showing the Oversight Agent catching deception.

# %%
from rich.console import Console
from rich.panel import Panel

console = Console()

# Turn on fast inference optimizations
FastLanguageModel.for_inference(rl_model.pretrained_model)

console.print("\n[bold magenta]--- FINAL VALIDATION EPISODE ---[/]")

obs, _ = env_wrapper.env.reset()
prompt_text = env_wrapper.generate_prompt(obs)

console.print(Panel(obs.get('claim_text', ''), title="[bold red]Malicious Sub-Agent Claim[/]"))

inputs = tokenizer(prompt_text, return_tensors="pt").to(rl_model.device)
outputs = rl_model.pretrained_model.generate(**inputs, max_new_tokens=128)

final_response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt_text, "")

# Ensure the agent actually output the <verdict> tags
action = env_wrapper.decode_action(final_response)
_, final_reward, _, _, _ = env_wrapper.env.step(action)

console.print(Panel(final_response, title="[bold cyan]Trained Oversight Agent Thought Process[/]"))
console.print(f"[bold green]Final Reward Assigned by Environment: {final_reward}[/]")
