import pandas as pd
import matplotlib.pyplot as plt

def plot_inventory_vs_demand(df):
    """Plot inventory levels vs demand over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['day'], df['inventory_level'], label='Inventory Level', color='blue')
    plt.plot(df['day'], df['demand'], label='Demand', color='orange')
    plt.axhline(y=10, color='red', linestyle='--', label='Min Inventory')
    plt.scatter(df[df['replenishment'] > 0]['day'], 
                df[df['replenishment'] > 0]['inventory_level'], 
                color='green', label='Replenishment', marker='^')
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.title('Inventory Levels vs Demand')
    plt.legend()
    plt.grid()
    plt.show()

def plot_transport_costs(df):
    """Plot transport costs over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['day'], df['transport_cost'], label='Transport Cost', color='purple', linestyle='--')
    plt.xlabel('Day')
    plt.ylabel('Cost')
    plt.title('Transport Costs Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def plot_rewards(file_path, title):
    """Plot rewards over training/testing episodes."""
    rewards = pd.read_csv(file_path, header=None)
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Cumulative Reward', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load the processed supply chain data
    data_file = "data/supply_chain_data_with_distances.csv"
    df = pd.read_csv(data_file)

    # Plot inventory vs demand
    plot_inventory_vs_demand(df)

    # Plot transport costs
    plot_transport_costs(df)

    # Plot rewards
    plot_rewards("models/training_rewards.csv", "Training Rewards Over Episodes")
    plot_rewards("models/testing_rewards.csv", "Testing Reward for Single Run")
