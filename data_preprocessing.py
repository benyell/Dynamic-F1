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
    df["stockout"] = df["remaining_inventory"].apply(lambda x: 1 if x < 0 else 0)
    df["overstock"] = df["inventory_level"].apply(lambda x: 1 if x > 50 else 0)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

    return df

# Function to visualize data
def visualize_data(df):
    print("Visualizing data...")
    plt.figure(figsize=(12, 6))

    # Plot inventory and demand
    plt.plot(df["day"], df["inventory_level"], label="Inventory Level", color="blue")
    plt.plot(df["day"], df["demand"], label="Demand", color="orange")
    plt.axhline(y=10, color="red", linestyle="--", label="Min Inventory")

    # Highlight replenishment days
    replenishment_days = df[df["replenishment"] > 0]["day"]
    plt.scatter(replenishment_days, df[df["replenishment"] > 0]["inventory_level"], color="green", label="Replenishment")

    # Add secondary y-axis for transport cost
    ax2 = plt.gca().twinx()
    ax2.plot(df["day"], df["transport_cost"], label="Transport Cost", color="purple", linestyle="--")
    ax2.set_ylabel("Transport Cost")
    ax2.legend(loc="upper right")

    # Add stockout and overstock markers
    stockout_days = df[df["remaining_inventory"] < 0]["day"]
    overstock_days = df[df["inventory_level"] > 50]["day"]
    plt.scatter(stockout_days, [10] * len(stockout_days), color="red", label="Stockouts", marker="x")
    plt.scatter(overstock_days, [50] * len(overstock_days), color="purple", label="Overstock", marker="o")

    # Add labels and legend
    plt.legend(loc="upper left")
    plt.title("Inventory and Demand Over Time")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.show()

# Function to calculate cumulative metrics
def calculate_metrics(df):
    stockouts = df["stockout"].sum()
    overstock_days = df["overstock"].sum()
    total_transport_cost = df["transport_cost"].sum()

    # Additional KPIs
    service_level = 1 - (stockouts / len(df))  # % of days without stockouts
    avg_inventory = df["inventory_level"].mean()

    print(f"Metrics Report:")
    print(f"Total Stockouts: {stockouts}")
    print(f"Total Overstock Days: {overstock_days}")
    print(f"Total Transportation Cost: {total_transport_cost:.2f}")
    print(f"Service Level: {service_level:.2%}")
    print(f"Average Inventory Level: {avg_inventory:.2f}")

if __name__ == "__main__":
    input_file = "data/supply_chain_data_with_telemetry.csv"
    output_file = "data/processed_supply_chain_data.csv"

    # Preprocess and visualize data
    processed_data = preprocess_data(input_file, output_file)
    visualize_data(processed_data)

    # Calculate and print metrics
    calculate_metrics(processed_data)
