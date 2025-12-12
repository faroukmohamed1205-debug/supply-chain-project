from stable_baselines3 import PPO

# Load the model
model = PPO.load("ppo_supply_chain")

# Print the Neural Network Architecture
print("\n--- Model Architecture (The Brain) ---")
print(model.policy)
print("--------------------------------------\n")