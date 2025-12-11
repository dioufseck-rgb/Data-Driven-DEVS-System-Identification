import pandas as pd
import numpy as np

def generate_power_plant_data(n_steps=3000, output_file='data/synthetic_plant.csv'):
    print("Generating Richer Training Data (Frequent Mode Switches)...")
    
    np.random.seed(42) # Fixed seed for reproducibility
    time = np.linspace(0, 300, n_steps)
    
    # 1. Fuel Input (Varies continuously)
    fuel_input = 0.5 + 0.3 * np.sin(time/20) + np.random.normal(0, 0.02, n_steps)
    fuel_input = np.clip(fuel_input, 0, 1)

    # 2. Categorical Input (Varies frequently for better learning)
    modes = []
    mode_multipliers = []
    
    current_mode = "ECO"
    current_mult = 1.0
    steps_until_switch = 0
    
    for t in range(n_steps):
        if steps_until_switch <= 0:
            # Pick a new random mode
            rand = np.random.rand()
            if rand < 0.33:
                current_mode = "OFF"
                current_mult = 0.0
            elif rand < 0.66:
                current_mode = "TURBO"
                current_mult = 2.0
            else:
                current_mode = "ECO"
                current_mult = 1.0
            
            # Hold this mode for a random duration (20 to 60 steps)
            steps_until_switch = np.random.randint(20, 60)
        
        modes.append(current_mode)
        mode_multipliers.append(current_mult)
        steps_until_switch -= 1

    # 3. Subsystem A: Boiler Temp
    boiler_temp = np.zeros(n_steps)
    boiler_temp[0] = 25.0
    
    for t in range(1, n_steps):
        # Physics: Target = Fuel * ModeMultiplier * MaxTemp
        target_temp = (fuel_input[t-1] * 600) * mode_multipliers[t-1]
        
        # In OFF mode, target is ambient
        if mode_multipliers[t-1] == 0:
            target_temp = 25.0
            
        # First order lag
        boiler_temp[t] = boiler_temp[t-1] + 0.05 * (target_temp - boiler_temp[t-1]) + np.random.normal(0, 0.5)

    # 4. Subsystem B: Turbine RPM
    turbine_rpm = np.zeros(n_steps)
    for t in range(1, n_steps):
        target_rpm = 0
        if boiler_temp[t-1] > 100:
            target_rpm = (boiler_temp[t-1] - 100) * 10 
        
        turbine_rpm[t] = turbine_rpm[t-1] + 0.02 * (target_rpm - turbine_rpm[t-1]) + np.random.normal(0, 2.0)

    # Save
    df = pd.DataFrame({
        'timestamp': time,
        'fuel_valve': fuel_input,
        'operation_mode': modes,
        'boiler_temp': boiler_temp,
        'turbine_rpm': turbine_rpm
    })
    
    df.to_csv(output_file, index=False)
    print(f"✅ Data generated. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    generate_power_plant_data()