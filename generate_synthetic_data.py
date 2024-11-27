import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Function to generate supply chain data with Fast F1 telemetry
def generate_supply_chain_data(race_schedule_file, telemetry_folder, num_days, output_file):
    print("Generating supply chain data with Fast F1 telemetry...")

    # Load race schedule
    race_schedule = pd.read_csv(race_schedule_file)
    race_schedule['EventDate'] = pd.to_datetime(race_schedule['EventDate'])

    # Simulate daily data
    start_date = race_schedule['EventDate'].min()
    end_date = start_date + timedelta(days=num_days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    data = {
        "day": range(1, num_days + 1),
        "date": dates,
        "inventory_level": np.random.randint(40, 60, num_days),
        "demand": np.random.randint(1, 5, num_days),  # Add random baseline demand
        "lead_time": np.random.randint(2, 5, num_days),
        "transport_cost": np.random.uniform(50, 150, num_days),
    }
    df = pd.DataFrame(data)

    # Add race-related demand spikes using telemetry
    for _, race in race_schedule.iterrows():
        race_date = race['EventDate']
        event_name = race['EventName']
        telemetry_file = os.path.join(telemetry_folder, f"telemetry_{event_name.replace(' ', '_')}_pitstops.csv")

        if os.path.exists(telemetry_file):
            pit_stops = pd.read_csv(telemetry_file)
            total_pit_stops = pit_stops['PitStopCount'].sum()  # Total pit stops in the race
            if race_date in df['date'].values:
                idx = df[df['date'] == race_date].index[0]
                spike = total_pit_stops * np.random.randint(2, 5)
                df.loc[idx, 'demand'] += spike  # Scale demand by pit stops
                print(f"Added demand spike of {spike} on {race_date} for {event_name} ({total_pit_stops} pit stops).")
        else:
            print(f"Telemetry file not found for {event_name}.")

    # Simulate inventory and replenishment
    min_inventory = 10
    replenishment_amount = 50
    df["remaining_inventory"] = df["inventory_level"] - df["demand"]
    df["replenishment"] = df["remaining_inventory"].apply(
        lambda x: replenishment_amount if x < min_inventory else 0
    )

    # Simulate delayed replenishment
    df["delayed_replenishment"] = 0
    for i in range(len(df)):
        lead_time = df.loc[i, "lead_time"]
        if i + lead_time < len(df):
            df.loc[i + lead_time, "delayed_replenishment"] += df.loc[i, "replenishment"]

    df["inventory_level"] = df["remaining_inventory"] + df["delayed_replenishment"]

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Supply chain data saved to {output_file}")

if __name__ == "__main__":
    race_schedule_file = "data/f1_race_schedule.csv"
    telemetry_folder = "data"
    output_file = "data/supply_chain_data_with_telemetry.csv"
    generate_supply_chain_data(race_schedule_file, telemetry_folder, 30, output_file)
