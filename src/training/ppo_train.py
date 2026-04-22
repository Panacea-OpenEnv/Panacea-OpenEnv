import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from unsloth import FastLanguageModel
import sys
import os

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.env import PanaceaEnv

def train_ppo(model_name="unsloth/llama-3-8b-bnb-4bit"):
    """
    Minimal PPO Training script boilerplate using Unsloth and TRL
    """
    print("Loading Unsloth FastLanguageModel...")
    max_seq_length = 2048 
    
    # NOTE: Uncomment running locally with GPU in later phases.
    # We are putting placeholders since this is the MVE initialization phase
    
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = model_name,
    #     max_seq_length = max_seq_length,
    #     dtype = None,
    #     load_in_4bit = True,
    # )
    
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 16, 
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_alpha = 16,
    #     lora_dropout = 0,
    #     bias = "none",
    #     use_gradient_checkpointing = "unsloth",
    #     random_state = 3407,
    #     use_rslora = False,
    # )

    # ppo_config = PPOConfig(
    #     batch_size=1, forward_batch_size=1
    # )

    env = PanaceaEnv()
    print("Environment initialized.")

    print("Training loop setup complete. Use `trl` PPOTrainer logic here to step the env.")
    
    # Typical execution:
    # obs, info = env.reset()
    # while not done:
    #     action = model_generate(obs) 
    #     next_obs, reward, done, truncated, info = env.step(action)
    #     ... push to replay buffer, train...

if __name__ == "__main__":
    train_ppo()
