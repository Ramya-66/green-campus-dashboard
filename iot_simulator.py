import pandas as pd
import random
import time
from datetime import datetime
import os

# List of buildings
buildings = ['Main Hall', 'Library', 'Cafeteria', 'Admin Block', 'Lab Building']

# Output file
csv_file = "live_data.csv"

# Initialize the CSV with headers
def initialize_csv():
    if not os.path.exists(csv_file):  # Only create if it doesn't exist
        df = pd.DataFrame(columns=['Timestamp', 'Building', 'Energy_kWh', 'Water_Liters', 'Occupancy'])
        df.to_csv(csv_file, index=False)
        print("📝 CSV initialized with headers.")
    else:
        print("✅ CSV already exists. Appending new data.")

# Generate a single row of fake data
def generate_data():
    return {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Building': random.choice(buildings),
        'Energy_kWh': round(random.uniform(20.0, 100.0), 2),
        'Water_Liters': round(random.uniform(200.0, 1000.0), 2),
        'Occupancy': random.randint(10, 100)
    }

# Append data to the CSV file every few seconds
def run_simulator():
    print("🌱 Starting IoT data simulator for Green Campus...")
    while True:
        data = generate_data()
        df = pd.DataFrame([data])
        df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
        print(f"✅ Data added: {data}")
        time.sleep(5)  # Every 5 seconds

if __name__ == "__main__":
    initialize_csv()
    run_simulator()
