import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.rl_environment import SupplyChainEnvironment
from rl_agent import RLAgent

import logging

logging.basicConfig(level=logging.INFO)

def test_agent(env, agent):
    state = env.reset()
    total_reward = 0
    done = False

    # List to store inventory levels and decisions
    results = []

    while not done:
        # Log current inventory level
        current_inventory = env.inventory_level

        # RL agent chooses an action
        action = agent.choose_action(state)

        # Apply the action and get the next state
        state, reward, done = env.step(action)

        # Log inventory level after action
        updated_inventory = env.inventory_level

        # Append the result
        results.append({
            "Day": env.current_day,
            "Action Taken (Inventory Adjustment)": action,
            "Inventory Before": current_inventory,
            "Inventory After": updated_inventory,
            "Reward": reward
        })

        total_reward += reward

    logging.info(f"Test Run Complete. Total Reward: {total_reward}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    max_inventory = 500
    max_days = 180

    env = SupplyChainEnvironment(
        data_file="data/final_processed_data.csv",
        distances_file="data/constructors_distances.csv",
        max_days=max_days,
        max_inventory=max_inventory,
    )

    agent = RLAgent.load_model("models/trained_rl_agent.pkl")
    if not agent:
        logging.error("Trained agent file not found. Please train the agent first.")
        sys.exit(1)

    test_results = test_agent(env, agent)
    test_results.to_csv("models/testing_results.csv", index=False)

    # # Plot Inventory Changes
    # plt.plot(test_results["Day"], test_results["Inventory Before"], label="Inventory Before Action")
    # plt.plot(test_results["Day"], test_results["Inventory After"], label="Inventory After Action")
    # plt.title("Inventory Levels Before and After RL Agent Actions")
    # plt.xlabel("Day")
    # plt.ylabel("Inventory Level")
    # plt.legend()
    # plt.show()
