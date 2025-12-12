import streamlit as st
import or_gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# --- Wrapper Class (Must allow loading the model correctly) ---
class OrGymWrapper(gym.Env):
    def __init__(self, env_id):
        self.env = or_gym.make(env_id)
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.array(obs, dtype=np.float32), float(reward), done, False, info

# --- Page Config ---
st.set_page_config(page_title="Supply Chain AI Optimization", layout="wide", page_icon="📦")

st.title("📦 Supply Chain Inventory Optimizer (RL Agent)")

# --- Sidebar ---
st.sidebar.header("⚙️ Simulation Settings")
days_to_simulate = st.sidebar.slider("Simulation Duration (Days)", 10, 60, 30)
run_btn = st.sidebar.button("🚀 Run Simulation")

# --- Load Model ---
MODEL_PATH = "ppo_supply_chain"
model = None

if run_btn:
    try:
        model = PPO.load(MODEL_PATH)
        st.sidebar.success("✅ Model Loaded")
    except:
        st.sidebar.error("⚠️ Model not found! Run training first.")

# --- Simulation Logic ---
def run_episode(agent_model=None):
    # Use the wrapper
    env = OrGymWrapper('InvManagement-v1')
    
    obs, _ = env.reset()
    
    inventory_levels = []
    rewards = []
    
    for _ in range(days_to_simulate):
        if agent_model:
            action, _ = agent_model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample() # Random action
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Access raw env state for visualization (Index 0 is typically Retailer Inv)
        retailer_inv = env.env.state[0]
        
        inventory_levels.append(retailer_inv)
        rewards.append(reward)
        
        if done:
            break
            
    return inventory_levels, rewards

# --- Main Execution ---
if run_btn and model:
    with st.spinner("Running Simulation..."):
        # AI Run
        ai_inv, ai_rewards = run_episode(model)
        # Random Run
        rnd_inv, rnd_rewards = run_episode(None)
        
        total_ai = sum(ai_rewards)
        total_rnd = sum(rnd_rewards)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🤖 AI Profit", f"${total_ai:,.0f}")
        col2.metric("🎲 Random Profit", f"${total_rnd:,.0f}")
        col3.metric("Difference", f"${total_ai - total_rnd:,.0f}")

        # Charts
        st.subheader("📊 Inventory Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ai_inv, name="AI Agent", line=dict(color='green')))
        fig.add_trace(go.Scatter(y=rnd_inv, name="Random", line=dict(color='red', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

elif run_btn and not model:
    st.warning("Please run `python train_agent.py` first!")