import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.rl_environment import SupplyChainEnvironment
from rl_agent import RLAgent

import logging

logging.basicConfig(level=logging.INFO)

def train_agent(env, agent, episodes):
    rewards_log = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_log.append(total_reward)
        logging.info(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

    logging.info("Training completed.")
    pd.Series(rewards_log).to_csv("models/training_rewards.csv", index=False)
    return agent


if __name__ == "__main__":
    max_inventory = 500
    max_days = 180

    env = SupplyChainEnvironment(
        data_file="data/final_processed_data.csv",
        distances_file="data/constructors_distances.csv",
        max_days=max_days,
        max_inventory=max_inventory,
    )
    state_size = max_days * max_inventory
    action_size = 11  # Actions: -5 to +5

    agent = RLAgent(state_size=state_size, action_size=action_size)
    trained_agent = train_agent(env, agent, episodes=1000)
    trained_agent.save_model("models/trained_rl_agent.pkl")
