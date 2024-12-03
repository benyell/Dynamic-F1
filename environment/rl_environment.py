import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SupplyChainEnvironment:
    def __init__(self, data_file, distances_file, max_days=50, max_inventory=500):
        self.data = pd.read_csv(data_file)
        self.distances = pd.read_csv(distances_file)

        # Initialize scalers
        self.scaler_demand = MinMaxScaler()
        self.scaler_transport = MinMaxScaler()
        self.scaler_inventory = MinMaxScaler(feature_range=(0, 1))  # Normalized to [0, 1]

        # Normalize data
        self.data["demand_normalized"] = self.scaler_demand.fit_transform(self.data[["demand"]])
        self.data["transport_cost_normalized"] = self.scaler_transport.fit_transform(self.data[["transport_cost"]])
        self.data["inventory_normalized"] = self.scaler_inventory.fit_transform(self.data[["inventory_level"]])

        self.max_days = max_days
        self.max_inventory = 1.0  # Normalized inventory range is [0, 1]
        self.current_day = 0
        self.inventory_level = self.data.iloc[0]["inventory_normalized"]  # Use normalized inventory
        self.state = None

    def encode_state(self, inventory_level, day):
        """
        Encodes the state as a single integer index.
        """
        inventory_index = min(max(int(inventory_level * 100), 0), 99)  # Scale to discrete range [0, 99]
        day_index = min(max(day, 0), self.max_days - 1)
        return day_index * 100 + inventory_index

    def reset(self):
        self.current_day = 0
        self.inventory_level = self.data.iloc[0]["inventory_normalized"]  # Use normalized inventory
        self.state = self.encode_state(self.inventory_level, self.current_day)
        return self.state

    def step(self, action):
        # Adjust inventory based on action (normalized action space)
        self.inventory_level += action / 100.0  # Actions are small adjustments (scaled to [0, 1])
        self.inventory_level = min(max(self.inventory_level, 0), self.max_inventory)

        # Get normalized demand and transport cost
        demand_normalized = self.data.iloc[self.current_day]["demand_normalized"]
        transport_cost_normalized = self.data.iloc[self.current_day]["transport_cost_normalized"]

        # Reward calculation
        stock_penalty = -50 if self.inventory_level < demand_normalized else 0
        reward = (self.inventory_level - demand_normalized) * 1 - transport_cost_normalized * 0.2 + stock_penalty

        # Update inventory after demand
        self.inventory_level = max(self.inventory_level - demand_normalized, 0)

        # Move to the next day
        self.current_day += 1
        done = self.current_day >= self.max_days

        if not done:
            # Use normalized inventory for the next day
            self.inventory_level = self.data.iloc[self.current_day]["inventory_normalized"]

        # Encode the new state
        self.state = self.encode_state(self.inventory_level, self.current_day)
        return self.state, reward, done
