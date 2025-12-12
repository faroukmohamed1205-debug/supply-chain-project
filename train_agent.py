import or_gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# --- Configuration ---
ENV_ID = 'InvManagement-v1'
TIMESTEPS = 2000000
MODEL_PATH = "ppo_supply_chain"
LOG_DIR = "logs"

# --- The Magic Wrapper (Manual Fix) ---
class OrGymWrapper(gym.Env):
    """
    A manual wrapper to convert old or-gym environments (Gym <=0.19) 
    to new Gymnasium environments (Gymnasium >=0.26) for SB3.
    """
    def __init__(self, env_id):
        self.env = or_gym.make(env_id)
        
        # 1. Convert Observation Space
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=np.float32
        )
        
        # 2. Convert Action Space
        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # Old gym doesn't take seed/options in reset
        obs = self.env.reset()
        # Ensure obs is float32
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # Old gym returns: obs, reward, done, info
        obs, reward, done, info = self.env.step(action)
        
        # New gymnasium expects: obs, reward, terminated, truncated, info
        return np.array(obs, dtype=np.float32), float(reward), done, False, info

def train_agent():
    # Create log dir
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"🔄 Initializing Environment: {ENV_ID}...")
    
    # Use our custom wrapper
    try:
        env = OrGymWrapper(ENV_ID)
        env = Monitor(env, LOG_DIR)
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    print("🧠 Setting up PPO Agent...")
    
    # Initialize Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01
    )

    print(f"🏋️ Starting Training for {TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TIMESTEPS)
        print("✅ Training Complete!")
        # Save the model
        model.save(MODEL_PATH)
        print(f"💾 Model saved to '{MODEL_PATH}.zip'")
    except Exception as e:
        print(f"❌ Training Interrupted: {e}")

if __name__ == "__main__":
    train_agent()
