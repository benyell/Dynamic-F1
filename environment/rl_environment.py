import numpy as np
import pandas as pd

class SupplyChainEnvironment:
    def __init__(self, data_file, distances_file):
        self.data = pd.read_csv(data_file)
        self.distances = pd.read_csv(distances_file)
        self.current_step = 0

        # State variables: inventory, demand, lead time, transport cost
        self.state = None
        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        self.current_step = 0
        self.state = self.data.iloc[self.current_step].to_dict()
        return self.state

    def step(self, action):
        """
        Take an action and move to the next state.
        Actions could be: reorder quantities or suppliers.
        """
        reward = self._calculate_reward(action)
        self.current_step += 1

        if self.current_step >= len(self.data):
            done = True
            next_state = None
        else:
            done = False
            next_state = self.data.iloc[self.current_step].to_dict()

        return next_state, reward, done

    def _calculate_reward(self, action):
        """Calculate the reward based on action and current state."""
        demand = self.state['demand']
        inventory = self.state['inventory_level']
        transport_cost = self.state['transport_cost']

        # Example reward function
        reward = -transport_cost
        if action > demand:
            reward -= (action - demand) * 2  # Overstock penalty
        elif action < demand:
            reward -= (demand - action) * 5  # Stockout penalty
        else:
            reward += 10  # Perfect match reward

        return reward
