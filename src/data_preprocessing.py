import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_files():
    # Define file paths
    pitstop_folder = "data"  # Folder containing telemetry_*_pitstops.csv files
    supply_chain_file = "data/supply_chain_data_with_distances.csv"
    race_schedule_file = "data/f1_race_schedule.csv"
    output_file = "data/final_processed_data.csv"

    # Load race schedule and supply chain data
    race_schedule = pd.read_csv(race_schedule_file)
    supply_chain_data = pd.read_csv(supply_chain_file)
    supply_chain_data["date"] = pd.to_datetime(supply_chain_data["date"])
    race_schedule["EventDate"] = pd.to_datetime(race_schedule["EventDate"])

    # Initialize MinMaxScaler for pit stop counts
    scaler = MinMaxScaler()

    # Aggregate all pit stop data
    pit_stop_data = []

    # Process all telemetry pit stop files in the specified folder
    for filename in os.listdir(pitstop_folder):
        if filename.startswith("telemetry_") and filename.endswith("_pitstops.csv"):
            file_path = os.path.join(pitstop_folder, filename)
            pit_stop_df = pd.read_csv(file_path)

            # Extract race name from file name
            race_name = filename.replace("telemetry_", "").replace("_pitstops.csv", "").replace("_", " ")

            # Normalize pit stop counts
            if "PitStopCount" in pit_stop_df.columns:
                pit_stop_df["NormalizedPitStops"] = scaler.fit_transform(pit_stop_df[["PitStopCount"]])

            # Summarize total normalized pit stops for the race
            total_pit_stops = pit_stop_df["NormalizedPitStops"].sum()
            event_date = race_schedule.loc[race_schedule["EventName"] == race_name, "EventDate"].values[0]

            # Append to aggregated data
            pit_stop_data.append({
                "EventName": race_name,
                "EventDate": event_date,
                "TotalNormalizedPitStops": total_pit_stops
            })

    # Create a DataFrame from aggregated pit stop data
    pit_stop_summary = pd.DataFrame(pit_stop_data)

    # Merge pit stop data with supply chain data
    supply_chain_data = supply_chain_data.merge(
        pit_stop_summary,
        how="left",
        left_on="date",
        right_on="EventDate"
    )

    # Fill missing values for races without telemetry data
    supply_chain_data["TotalNormalizedPitStops"] = supply_chain_data["TotalNormalizedPitStops"].fillna(0)

    # Step 1: Shift inventory levels to remove negative values
    offset = abs(supply_chain_data["remaining_inventory"].min()) + 1
    supply_chain_data["remaining_inventory"] += offset
    supply_chain_data["inventory_level"] += offset

    # Step 2: Create a 'shortage' column to capture inventory shortfalls
    supply_chain_data["shortage"] = supply_chain_data["demand"] - supply_chain_data["inventory_level"]
    supply_chain_data["shortage"] = supply_chain_data["shortage"].apply(lambda x: x if x > 0 else 0)

    # Normalize replenishment to handle wide range
    supply_chain_data["replenishment"] = supply_chain_data["replenishment"].clip(lower=0)

    # Step 3: Normalize relevant columns
    columns_to_normalize = ["inventory_level", "demand", "transport_cost", "shortage", "replenishment", "remaining_inventory"]
    supply_chain_data[columns_to_normalize] = scaler.fit_transform(supply_chain_data[columns_to_normalize])

    # Step 4: Save the final processed dataset
    supply_chain_data.to_csv(output_file, index=False)
    print(f"Final preprocessed data saved to {output_file}")


if __name__ == "__main__":
    preprocess_files()
