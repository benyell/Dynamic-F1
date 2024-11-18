import numpy as np
import pandas as pd
import os

# Function to generate synthetic supply chain data
def generate_synthetic_data(num_days, output_file):
    print("Generating synthetic supply chain data...")
    max_inventory = 100
    min_inventory = 10

    # Generate synthetic data
    data = {
        "day": range(1, num_days + 1),
        "inventory_level": np.random.randint(min_inventory, max_inventory, num_days),
        "demand": np.random.randint(5, 20, num_days),  # Random daily demand
        "lead_time": np.random.randint(2, 7, num_days),  # Random lead time in days
        "transport_cost": np.random.uniform(50, 200, num_days),  # Transport costs
    }

    # Save to CSV
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Synthetic data saved to {output_file}")

if __name__ == "__main__":
    output_file = "data/supply_chain_data.csv"
    generate_synthetic_data(30, output_file)
