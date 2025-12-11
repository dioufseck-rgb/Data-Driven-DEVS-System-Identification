import pandas as pd
import matplotlib.pyplot as plt
import os

def validate_and_save_plots(input_file='data/synthetic_plant.csv'):
    # 1. Check if data exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run generator.py first.")
        return

    # 2. Create 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Created 'plots' directory.")

    # 3. Load Data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # 4. Create Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot Input: Fuel
    axes[0].plot(df['timestamp'], df['fuel_valve'], color='blue', linewidth=1.5)
    axes[0].set_ylabel('Fuel Valve (0-1)')
    axes[0].set_title('System Input: Fuel Control')
    axes[0].grid(True, alpha=0.3)

    # Plot Subsystem A: Temp
    axes[1].plot(df['timestamp'], df['boiler_temp'], color='red', linewidth=1.5)
    axes[1].set_ylabel('Temperature (C)')
    axes[1].set_title('Subsystem A: Boiler Response (Lagged)')
    axes[1].grid(True, alpha=0.3)

    # Plot Subsystem B: RPM
    axes[2].plot(df['timestamp'], df['turbine_rpm'], color='green', linewidth=1.5)
    axes[2].set_ylabel('Turbine RPM')
    axes[2].set_title('Subsystem B: Turbine Response (Dependent on A)')
    axes[2].grid(True, alpha=0.3)

    # X-Axis Label
    plt.xlabel('Time (Simulation Steps)')
    plt.tight_layout()

    # 5. Save to File
    output_path = 'plots/system_validation.png'
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved successfully to: {output_path}")
    
    # Optional: Close plot to free memory
    plt.close()

if __name__ == "__main__":
    validate_and_save_plots()