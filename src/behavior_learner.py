import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# Try to import structure learner, exit if missing
try:
    from structure_learner import learn_structure
except ImportError:
    exit()

if not os.path.exists('models'):
    os.makedirs('models')

class SubsystemModel:
    def __init__(self, target_col, input_cols, feature_types):
        self.target_col = target_col
        self.input_cols = input_cols
        
        # --- BUILD PIPELINE ---
        cat_cols = feature_types['categorical']
        num_cols = feature_types['numeric']
        
        # FIX 1: sparse_output=False prevents matrix issues
        # FIX 2: handle_unknown='ignore' prevents crashes on new categories
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', num_cols),
                ('cat', categorical_transformer, cat_cols)
            ],
            verbose_feature_names_out=False
        )
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
    def prepare_data(self, df):
        # 1. Target (Current t)
        y = df[self.target_col].iloc[1:]
        
        # 2. Inputs (Previous t-1)
        lagged_df = df.shift(1).iloc[1:]
        
        X = pd.DataFrame()
        
        # Add Self (Autoregression)
        X[f'{self.target_col}_prev'] = lagged_df[self.target_col]
        
        # Add Inputs
        for col in self.input_cols:
            if col in df.columns:
                X[f'{col}_prev'] = lagged_df[col]
        
        # FIX 3: Fill NaNs (Shift creates them at row 0, but safety first)
        X = X.fillna(0) # Or appropriate default
        
        return X, y

    def train(self, df):
        print(f"   [Training] Target: {self.target_col} | Inputs: {self.input_cols}")
        
        # --- DYNAMIC TYPE DETECTION ---
        feature_types = {'numeric': [], 'categorical': []}
        
        # 1. Check Target Type (Autoregression)
        col_name = f'{self.target_col}_prev'
        if df[self.target_col].dtype == 'object':
            feature_types['categorical'].append(col_name)
        else:
            feature_types['numeric'].append(col_name)
        
        # 2. Check Input Types
        for col in self.input_cols:
            lag_name = f'{col}_prev'
            if col not in df.columns: continue
            
            if df[col].dtype == 'object':
                feature_types['categorical'].append(lag_name)
            else:
                feature_types['numeric'].append(lag_name)
        
        # Re-initialize with correct types
        self.__init__(self.target_col, self.input_cols, feature_types)
        
        # Prepare Data
        X, y = self.prepare_data(df)
        
        # If Target is Categorical, we can't use Regressor directly on y.
        # But for this specific Hybrid System, Targets (Temp, RPM) are numeric.
        # If 'operation_mode' was learned as a target, we skip training it to avoid error.
        if df[self.target_col].dtype == 'object':
            print(f"      [SKIP] Target '{self.target_col}' is categorical. Skipping regression.")
            return

        try:
            self.pipeline.fit(X, y)
            preds = self.pipeline.predict(X)
            print(f"      -> Accuracy (R2): {r2_score(y, preds):.4f}")
        except Exception as e:
            print(f"      [ERROR] Training failed for {self.target_col}: {e}")

    def save(self):
        # Only save if pipeline is fitted
        try:
            # Check if model is fitted
            self.pipeline.named_steps['model'].feature_importances_
            with open(f"models/model_{self.target_col}.pkl", 'wb') as f:
                pickle.dump(self, f)
        except:
            pass # Skip saving unfitted models

def run_auto_behavior_learning():
    data_path = 'data/synthetic_plant.csv'
    if not os.path.exists(data_path):
        print("Data file missing.")
        return
    
    df = pd.read_csv(data_path)
    
    print("\n--- PHASE 2: Structure ---")
    G = learn_structure(input_file=data_path)
    
    print("\n--- PHASE 3: Training with Mixed Types ---")
    
    if G is None: return

    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        
        # If no inputs, it's a global source, skip
        if not predecessors: 
            continue
        
        # Initialize and Train
        # (Pass dummy types initially, 'train' will fix them)
        model = SubsystemModel(node, predecessors, {'numeric':[], 'categorical':[]}) 
        model.train(df)
        model.save()
        
    print("\n✅ Behavioral Learning Complete.")

if __name__ == "__main__":
    run_auto_behavior_learning()