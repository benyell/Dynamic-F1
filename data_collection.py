import fastf1
import pandas as pd
import os

# Enable cache
fastf1.Cache.enable_cache('cache')

# Function to fetch race schedules for a season
def fetch_race_schedule(year):
    print(f"Fetching race schedule for {year}...")
    schedule = fastf1.get_event_schedule(year)
    races = schedule[['RoundNumber', 'EventName', 'Location', 'EventDate']]
    return races

# Function to fetch lap times for a specific session
def fetch_lap_times(year, event_name, session_type):
    print(f"Fetching {session_type} session data for {event_name} in {year}...")
    session = fastf1.get_session(year, event_name, session_type)
    session.load()
    lap_data = session.laps[['Driver', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']]
    return lap_data

# Main function to collect data
def collect_data():
    year = 2023  # Define the year of interest
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Fetch race schedule
    race_schedule = fetch_race_schedule(year)
    race_schedule.to_csv(os.path.join(output_dir, "race_schedule.csv"), index=False)
    print("Race schedule saved to data/race_schedule.csv")

    # Fetch lap times for a specific race
    lap_times = fetch_lap_times(year, "Bahrain", "Race")
    lap_times.to_csv(os.path.join(output_dir, "bahrain_lap_times.csv"), index=False)
    print("Lap times saved to data/bahrain_lap_times.csv")

if __name__ == "__main__":
    collect_data()
