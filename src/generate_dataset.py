import pandas as pd
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the geolocator
geolocator = Nominatim(user_agent="f1_supply_chain_optimizer")

# Function to add country names to manufacturers
def map_countries(manufacturers_file, countries_file, output_file):
    logging.info(f"Mapping countries for {manufacturers_file}...")
    manufacturers = pd.read_csv(manufacturers_file)
    countries = pd.read_csv(countries_file)

    # Merge with country data to get country names
    manufacturers = manufacturers.merge(countries[['id', 'name']], left_on='countryId', right_on='id', how='left')
    manufacturers.rename(columns={'name': 'Country'}, inplace=True)

    # Drop unnecessary columns
    manufacturers.drop(columns=['id_y'], inplace=True)

    # Save the updated file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    manufacturers.to_csv(output_file, index=False)
    logging.info(f"Mapped countries saved to {output_file}.")
    return manufacturers

# Function to add coordinates to manufacturers
def add_coordinates(manufacturers_file, output_file):
    logging.info(f"Adding coordinates for {manufacturers_file}...")
    manufacturers = pd.read_csv(manufacturers_file)
    manufacturers['Coordinates'] = None
    
    # Geocode each country
    for index, row in manufacturers.iterrows():
        try:
            logging.info(f"Geocoding country: {row['countryId']} ({index+1}/{len(manufacturers)})")
            location = geolocator.geocode(row['countryId'])
            if location:
                manufacturers.at[index, 'Coordinates'] = f"{location.latitude},{location.longitude}"
            else:
                logging.warning(f"Could not find coordinates for: {row['countryId']}")
        except Exception as e:
            logging.error(f"Error geocoding {row['countryId']}: {e}")
    
    # Save the updated file
    manufacturers.to_csv(output_file, index=False)
    logging.info(f"Coordinates added and saved to {output_file}.")
    return manufacturers

# Function to calculate distances between races and manufacturers
def calculate_distances(schedule_file, manufacturers_file, output_file):
    logging.info(f"Calculating distances from {manufacturers_file} to race locations in {schedule_file}...")
    race_schedule = pd.read_csv(schedule_file)
    manufacturers = pd.read_csv(manufacturers_file)
    
    # Extract coordinates
    race_schedule['Coordinates'] = race_schedule['Location'].apply(
        lambda loc: geolocator.geocode(loc) and (geolocator.geocode(loc).latitude, geolocator.geocode(loc).longitude)
    )
    manufacturers['Coordinates'] = manufacturers['Coordinates'].apply(
        lambda coord: tuple(map(float, coord.strip('()').split(','))) if pd.notna(coord) else None
    )
    
    results = []
    for race_index, race in race_schedule.iterrows():
        if not race['Coordinates']:
            logging.warning(f"Skipping race {race['EventName']} as it has no coordinates.")
            continue
        
        logging.info(f"Processing race: {race['EventName']} ({race_index+1}/{len(race_schedule)})")
        race_coords = race['Coordinates']
        race_result = {'Race': race['EventName']}
        
        for _, manufacturer in manufacturers.iterrows():
            manu_coords = manufacturer['Coordinates']
            if manu_coords:
                distance = geodesic(race_coords, manu_coords).km
                race_result[manufacturer['countryId']] = distance
        
        results.append(race_result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logging.info(f"Distance calculations saved to {output_file}.")
    return results_df

if __name__ == "__main__":
    logging.info("Starting F1 Supply Chain Optimization Script...")
    
    # File paths
    countries_file = "data/f1db/f1db-countries.csv"
    constructors_file = "data/f1db/f1db-constructors.csv"
    engine_manufacturers_file = "data/f1db/f1db-engine-manufacturers.csv"
    tyre_manufacturers_file = "data/f1db/f1db-tyre-manufacturers.csv"
    race_schedule_file = "data/f1_race_schedule.csv"
    
    # Output files
    constructors_with_countries = "data/constructors_with_countries.csv"
    engine_manufacturers_with_countries = "data/engine_manufacturers_with_countries.csv"
    tyre_manufacturers_with_countries = "data/tyre_manufacturers_with_countries.csv"
    constructors_with_coordinates = "data/constructors_with_coordinates.csv"
    engine_manufacturers_with_coordinates = "data/engine_manufacturers_with_coordinates.csv"
    tyre_manufacturers_with_coordinates = "data/tyre_manufacturers_with_coordinates.csv"
    constructors_distances_file = "data/constructors_distances.csv"
    engine_manufacturers_distances_file = "data/engine_manufacturers_distances.csv"
    tyre_manufacturers_distances_file = "data/tyre_manufacturers_distances.csv"
    
    # Process constructors
    map_countries(constructors_file, countries_file, constructors_with_countries)
    add_coordinates(constructors_with_countries, constructors_with_coordinates)
    calculate_distances(race_schedule_file, constructors_with_coordinates, constructors_distances_file)
    
    # Process engine manufacturers
    map_countries(engine_manufacturers_file, countries_file, engine_manufacturers_with_countries)
    add_coordinates(engine_manufacturers_with_countries, engine_manufacturers_with_coordinates)
    calculate_distances(race_schedule_file, engine_manufacturers_with_coordinates, engine_manufacturers_distances_file)
    
    # Process tyre manufacturers
    map_countries(tyre_manufacturers_file, countries_file, tyre_manufacturers_with_countries)
    add_coordinates(tyre_manufacturers_with_countries, tyre_manufacturers_with_coordinates)
    calculate_distances(race_schedule_file, tyre_manufacturers_with_coordinates, tyre_manufacturers_distances_file)
    
    logging.info("Script execution complete.")
