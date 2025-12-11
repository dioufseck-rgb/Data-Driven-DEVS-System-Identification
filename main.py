import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Import our generator
from src.generator import generate_power_plant_data

def run_pipeline():
    print("--- 1. INITIALIZATION ---")
    data_path = 'data/synthetic_plant.csv'
    
    # Check if data exists, if not, generate it
    if not os.path.exists(data_path):
        print("Data not found. Generating...")
        df = generate_power_plant_data()
        df.to_csv(data_path, index=False)
    else:
        print("Loading existing data...")
        df = pd.read_csv(data_path)
    
    print(f"Data Loaded. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Quick Visualization check
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(df['fuel_valve'], color='orange')
    plt.title("Input: Fuel Valve")
    
    plt.subplot(3, 1, 2)
    plt.plot(df['boiler_temp'], color='red')
    plt.title("Subsystem A: Boiler Temp")
    
    plt.subplot(3, 1, 3)
    plt.plot(df['turbine_rpm'], color='green')
    plt.title("Subsystem B: Turbine RPM")
    
    plt.tight_layout()
    plt.show()

    print("\n✅ Phase 1 Complete: Environment Ready.")

if __name__ == "__main__":
    run_pipeline()