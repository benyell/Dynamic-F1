import fastf1
import pandas as pd
import os

# Enable Fast F1 cache
fastf1.Cache.enable_cache('cache')

# Function to fetch race schedule for a given year
def fetch_race_schedule(year, output_file):
    try:
        print(f"Fetching race schedule for {year}...")
        schedule = fastf1.get_event_schedule(year)
        races = schedule[['RoundNumber', 'EventName', 'Location', 'EventDate']]
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        races.to_csv(output_file, index=False)
        print(f"Race schedule saved to {output_file}")
    except Exception as e:
        print(f"Failed to fetch race schedule for {year}: {e}")

def fetch_race_telemetry(year, event_name, output_file):
    try:
        print(f"Fetching telemetry for {event_name} ({year})...")
        session = fastf1.get_session(year, event_name, 'R')  # Fetch race session by name
        session.load()  # Load telemetry data

        # Example: Inspect results to find available columns
        print("Available columns in session results:", session.results.columns)

        # Calculate pit stops from lap data
        laps = session.laps
        pit_stops = laps[laps['PitOutTime'].notna() | laps['PitInTime'].notna()]
        pit_stops_count = pit_stops.groupby('Driver')['PitOutTime'].count().reset_index()
        pit_stops_count.rename(columns={'PitOutTime': 'PitStopCount'}, inplace=True)

        # Save lap and pit stop data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        laps.to_csv(output_file.replace('.csv', '_laps.csv'), index=False)
        pit_stops_count.to_csv(output_file.replace('.csv', '_pitstops.csv'), index=False)
        print(f"Telemetry data and calculated pit stops saved for {event_name}")
    except Exception as e:
        print(f"Failed to fetch telemetry for {event_name}: {e}")




if __name__ == "__main__":
    # Fetch schedule for the 2023 season
    schedule_file = "data/f1_race_schedule.csv"
    fetch_race_schedule(2023, schedule_file)

    # Fetch telemetry for each race in the schedule
    race_schedule = pd.read_csv(schedule_file)
    # Filter out non-race events like pre-season testing
    race_schedule = race_schedule[race_schedule['EventName'] != "Pre-Season Testing"]

    # Fetch telemetry for each race in the schedule
    for _, race in race_schedule.iterrows():
        event_name = race['EventName']
        telemetry_file = f"data/telemetry_{event_name.replace(' ', '_')}.csv"
        fetch_race_telemetry(2023, event_name, telemetry_file)