import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Try to import structure learner
try:
    from structure_learner import learn_structure
except ImportError:
    print("Warning: structure_learner.py not found.")

# --- CLASS DEFINITION FOR PICKLE LOADING ---
# This must match behavior_learner.py exactly
class SubsystemModel:
    def __init__(self, target_col, input_cols, feature_types):
        self.target_col = target_col
        self.input_cols = input_cols
        self.pipeline = None # Real object loads this from pickle

# --- DEVS BLOCK ---
class DEVS_Block:
    def __init__(self, node_name):
        self.name = node_name
        self.model_obj = self.load_model()
        self.state = 0.0 
        self.next_state = 0.0
        
    def load_model(self):
        path = f"models/model_{self.name}.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def predict_next_step(self, inputs_dict):
        """
        Predicts state(t) based on state(t-1) + inputs(t-1).
        Constructs a DataFrame so the Pipeline can handle categorical encoding.
        """
        if self.model_obj is None:
            return self.state

        # Construct Input Data Dictionary
        # Keys must match the training features: '{name}_prev'
        input_data = {}
        
        # 1. Autoregression (Self Previous)
        input_data[f'{self.name}_prev'] = [self.state]
        
        # 2. Exogenous Inputs (Neighbors Previous)
        for input_name in self.model_obj.input_cols:
            val = inputs_dict.get(input_name, 0.0) 
            input_data[f'{input_name}_prev'] = [val]
            
        # Convert to DataFrame (Essential for Pipeline column names)
        X_pred = pd.DataFrame(input_data)
        
        try:
            # The Pipeline handles OneHotEncoding internally!
            self.next_state = self.model_obj.pipeline.predict(X_pred)[0]
        except Exception as e:
            # Fallback for errors (e.g., first step alignment)
            # print(f"Error predicting {self.name}: {e}")
            self.next_state = self.state 

    def commit(self):
        self.state = self.next_state
        return self.state

# --- SIMULATION ENGINE ---
class SimulationEngine:
    def __init__(self, data_file='data/synthetic_plant.csv'):
        print("--- Initializing Simulator ---")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file '{data_file}' not found. Run generator.py first.")

        # 1. Get Structure
        self.G = learn_structure(data_file)
        if self.G is None:
            raise ValueError("Could not learn structure.")

        self.blocks = {}
        
        # 2. Instantiate Blocks
        for node in self.G.nodes():
            path = f"models/model_{node}.pkl"
            if os.path.exists(path):
                print(f"   [LOADED] Model for: {node}")
                self.blocks[node] = DEVS_Block(node)
            else:
                print(f"   [SOURCE] External Input: {node}")

    def run(self, input_scenario_df, initial_conditions=None):
        print(f"\n--- Running Simulation ({len(input_scenario_df)} steps) ---")
        
        history = []
        
        # 1. Set Initial Conditions
        if initial_conditions:
            for node, val in initial_conditions.items():
                if node in self.blocks:
                    self.blocks[node].state = val

        # 2. Time Loop
        for t in range(len(input_scenario_df)):
            
            # A. Get Inputs for this step
            current_inputs = input_scenario_df.iloc[t].to_dict()
            
            # --- HARD OVERRIDE ---
            # If the user provided a column (e.g., 'fuel_valve' or 'operation_mode'),
            # force that value into the system state.
            for col_name, val in current_inputs.items():
                if col_name in self.blocks:
                    self.blocks[col_name].state = val
                    self.blocks[col_name].next_state = val 
            
            # B. Snapshot System State (t-1)
            system_snapshot = current_inputs.copy()
            for name, block in self.blocks.items():
                if name not in system_snapshot:
                    system_snapshot[name] = block.state
            
            # C. Predict Next Step (t)
            # Only predict variables NOT provided in the scenario
            for name, block in self.blocks.items():
                if name not in input_scenario_df.columns:
                    block.predict_next_step(system_snapshot)
                
            # D. Commit
            step_record = {'time': t}
            step_record.update(current_inputs) 
            
            for name, block in self.blocks.items():
                if name not in input_scenario_df.columns:
                    step_record[name] = block.commit()
                else:
                    step_record[name] = block.state
            
            history.append(step_record)

        return pd.DataFrame(history)

# --- EXECUTION ---
if __name__ == "__main__":
    # --- HYBRID SCENARIO DEFINITION ---
    steps = 300
    
    # 1. Fuel Input (Numeric): Keep it constant to isolate Mode effect
    fuel = np.array([0.5] * steps)
    
    # 2. Mode Input (Categorical): Switch modes over time
    # 0-100:   ECO (Normal)
    # 100-200: TURBO (High Temp)
    # 200-300: OFF (Cooldown)
    modes = ['ECO'] * steps
    for i in range(100, 200): modes[i] = 'TURBO'
    for i in range(200, 300): modes[i] = 'OFF'
    
    scenario_df = pd.DataFrame({
        'fuel_valve': fuel,
        'operation_mode': modes
    })

    # Run Simulation
    sim = SimulationEngine('data/synthetic_plant.csv')
    init_cond = {'boiler_temp': 25.0, 'turbine_rpm': 0.0}
    
    results = sim.run(scenario_df, init_cond)
    
    # --- PLOTTING ---
    if not results.empty:
        plt.figure(figsize=(10, 10))
        
        # Plot 1: Inputs (Visualizing Modes)
        plt.subplot(3, 1, 1)
        plt.plot(results['fuel_valve'], 'b', label='Fuel Level')
        plt.title("Scenario Inputs: Fuel (Constant) + Modes (Changing)")
        plt.ylim(0, 1.1)
        
        # Add colored backgrounds for modes
        plt.axvspan(0, 100, color='green', alpha=0.1, label='ECO')
        plt.axvspan(100, 200, color='yellow', alpha=0.3, label='TURBO')
        plt.axvspan(200, 300, color='gray', alpha=0.3, label='OFF')
        plt.legend(loc='upper right')
        
        # Plot 2: Boiler
        if 'boiler_temp' in results.columns:
            plt.subplot(3, 1, 2)
            plt.plot(results['boiler_temp'], 'r', linewidth=2)
            plt.title("Predicted Boiler Temp (Reacts to Mode Changes!)")
            plt.grid(True, alpha=0.3)
            # Re-add background for context
            plt.axvspan(100, 200, color='yellow', alpha=0.1)
            plt.axvspan(200, 300, color='gray', alpha=0.1)
            
        # Plot 3: Turbine
        if 'turbine_rpm' in results.columns:
            plt.subplot(3, 1, 3)
            plt.plot(results['turbine_rpm'], 'g', linewidth=2)
            plt.title("Predicted Turbine RPM")
            plt.grid(True, alpha=0.3)
            plt.axvspan(100, 200, color='yellow', alpha=0.1)
            plt.axvspan(200, 300, color='gray', alpha=0.1)

        plt.tight_layout()
        
        if not os.path.exists('plots'): os.makedirs('plots')
        plt.savefig('plots/simulation_hybrid.png')
        print("✅ Hybrid Simulation Complete. Check plots/simulation_hybrid.png")