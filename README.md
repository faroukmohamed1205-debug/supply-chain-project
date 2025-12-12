# 📦 AI-Driven Supply Chain Inventory Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-PPO-green)
![Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📖 Project Overview
Inventory management is a complex balancing act: holding too much stock incurs high storage costs, while holding too little leads to stockouts and lost customers. Traditional methods (like Min-Max or EOQ) often struggle with stochastic demand and multi-echelon complexities.

This project implements a **Deep Reinforcement Learning (DRL)** agent using the **PPO (Proximal Policy Optimization)** algorithm to master this balance autonomously. The agent learns a **Just-In-Time (JIT)** strategy, optimizing reorder levels across a multi-stage supply chain to maximize net profit.

## 🧠 Key Features
* **Multi-Echelon Environment:** Simulates a realistic supply chain (Retailer → Wholesaler → Distributor → Manufacturer) using `or-gym` (`InvManagement-v1`).
* **State-of-the-Art RL:** Uses **PPO** from `stable-baselines3` for stable and efficient policy learning.
* **Interactive Dashboard:** A professional Streamlit app to visualize the agent's performance vs. a baseline random policy.
* **Performance Metrics:** Real-time tracking of Profit, Inventory Levels, and Stockouts.
* **Robust Engineering:** Custom wrappers to handle compatibility between legacy `Gym` environments and modern `Gymnasium` interfaces.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **ML Libraries:** Stable-Baselines3, NumPy, Pandas, PyTorch
* **Environment:** OR-Gym (Operations Research Gym)
* **Visualization:** Streamlit, Plotly Interactive Charts

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd supply_chain_project
Install dependencies:Bashpip install -r requirements.txt
Train the Agent:To train the PPO model from scratch (Default: 200,000 timesteps):Bashpython train_agent.py
This will generate a ppo_supply_chain.zip file.Run the Dashboard:To visualize the results and compare AI vs. Random decisions:Bashpython -m streamlit run app.py
📊 Results & AnalysisThe "Aha!" MomentInitially, the agent fell into a local minimum (The "Lazy Agent" problem), reducing inventory to zero to avoid holding costs, resulting in a net loss. After increasing training to 200k steps, the agent learned that Sales Revenue > Holding Costs.Performance Comparison (30-Day Simulation)MetricRandom Policy 🎲PPO AI Agent 🤖ImprovementTotal Profit~$65**~$329**+400%Inventory StrategyErratic & OverstockedLean & Just-In-TimeOptimizedObservation: The AI agent maintains a low but sufficient inventory level, reacting dynamically to demand spikes, whereas the random policy fluctuates between overstocking (high cost) and stockouts (lost sales).(Place your screenshot here, e.g., )📂 Project StructureBash├── train_agent.py       # Main script to train and save the PPO model
├── app.py               # Streamlit dashboard for visualization & comparison
├── requirements.txt     # List of dependencies
├── ppo_supply_chain.zip # The trained model (Neural Network weights)
└── README.md            # Project documentation
🔮 Future ImprovementsHyperparameter Tuning: Use Optuna to find the perfect learning rate and batch size.Demand Forecasting: Integrate an LSTM layer to predict future demand based on history.Deployment: Containerize the application using Docker for cloud deployment.🤝 AuthorFarouk Mohamed FaroukFaculty of Engineering, AI MajorNew Ismailia National University