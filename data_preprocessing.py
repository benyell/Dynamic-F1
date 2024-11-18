import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to preprocess data
def preprocess_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Remove invalid rows
    print("Cleaning data...")
    df = df[df["inventory_level"] >= 0]

    # Add calculated features
    print("Adding features...")
    df["remaining_inventory"] = df["inventory_level"] - df["demand"]
    df["stock_status"] = df["remaining_inventory"].apply(
        lambda x: "Stockout" if x < 0 else ("Overstock" if x > 50 else "Normal")
    )

    # Save cleaned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

    return df

# Function to visualize data
def visualize_data(df):
    print("Visualizing data...")
    plt.figure(figsize=(10, 6))
    plt.plot(df["day"], df["inventory_level"], label="Inventory Level")
    plt.plot(df["day"], df["demand"], label="Demand")
    plt.axhline(y=10, color='r', linestyle='--', label="Min Inventory")
    plt.legend()
    plt.title("Inventory and Demand Over Time")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.show()

if __name__ == "__main__":
    input_file = "data/supply_chain_data.csv"
    output_file = "data/processed_supply_chain_data.csv"
    processed_data = preprocess_data(input_file, output_file)
    visualize_data(processed_data)
