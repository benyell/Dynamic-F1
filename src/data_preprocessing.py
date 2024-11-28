import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to preprocess data
def preprocess_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Flag negative inventory levels
    print("Flagging stockouts and inefficiencies...")
    df["stockout"] = df["inventory_level"].apply(lambda x: 1 if x < 0 else 0)
    df["excessive_transport_cost"] = df["transport_cost"].apply(lambda x: 1 if x > 200 else 0)
    df["excessive_lead_time"] = df["lead_time"].apply(lambda x: 1 if x > 7 else 0)

    # Add calculated features
    print("Adding features...")
    df["remaining_inventory"] = df["inventory_level"] - df["demand"]
    df["overstock"] = df["inventory_level"].apply(lambda x: 1 if x > 50 else 0)

    # Save processed data
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

    # Add stockout, overstock, and inefficiency markers
    stockout_days = df[df["stockout"] == 1]["day"]
    overstock_days = df[df["overstock"] == 1]["day"]
    excessive_cost_days = df[df["excessive_transport_cost"] == 1]["day"]
    excessive_lead_time_days = df[df["excessive_lead_time"] == 1]["day"]

    plt.scatter(stockout_days, [10] * len(stockout_days), color="red", label="Stockouts", marker="x")
    plt.scatter(overstock_days, [50] * len(overstock_days), color="purple", label="Overstock", marker="o")
    plt.scatter(excessive_cost_days, [200] * len(excessive_cost_days), color="brown", label="High Transport Cost", marker="D")
    plt.scatter(excessive_lead_time_days, [200] * len(excessive_lead_time_days), color="cyan", label="High Lead Time", marker="s")

    # Add labels and legend
    plt.legend(loc="upper left")
    plt.title("Supply Chain Metrics Over Time")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.show()

# Function to calculate cumulative metrics
def calculate_metrics(df):
    stockouts = df["stockout"].sum()
    overstock_days = df["overstock"].sum()
    excessive_transport_cost_days = df["excessive_transport_cost"].sum()
    excessive_lead_time_days = df["excessive_lead_time"].sum()
    total_transport_cost = df["transport_cost"].sum()

    # Additional KPIs
    service_level = 1 - (stockouts / len(df))  # % of days without stockouts
    avg_inventory = df["inventory_level"].mean()

    print(f"Metrics Report:")
    print(f"Total Stockouts: {stockouts}")
    print(f"Total Overstock Days: {overstock_days}")
    print(f"Excessive Transport Cost Days: {excessive_transport_cost_days}")
    print(f"Excessive Lead Time Days: {excessive_lead_time_days}")
    print(f"Total Transport Cost: {total_transport_cost:.2f}")
    print(f"Service Level: {service_level:.2%}")
    print(f"Average Inventory Level: {avg_inventory:.2f}")

if __name__ == "__main__":
    input_file = "data/supply_chain_data_with_distances.csv"
    output_file = "data/processed_supply_chain_data.csv"

    # Preprocess and visualize data
    processed_data = preprocess_data(input_file, output_file)
    visualize_data(processed_data)

    # Calculate and print metrics
    calculate_metrics(processed_data)
