import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import LabelEncoder
import os

def learn_structure(input_file='data/synthetic_plant.csv', excluded_cols=None, max_lag=5, p_value_threshold=0.05):
    # Note: Increased p_value_threshold to 0.05 to be more sensitive
    
    if excluded_cols is None:
        excluded_cols = ['timestamp', 'time', 'date', 'id']

    if not os.path.exists(input_file):
        return None
    
    df = pd.read_csv(input_file)
    analysis_df = df.copy()
    
    # Drop Excluded
    cols_to_drop = [c for c in analysis_df.columns if c.lower() in excluded_cols]
    analysis_df.drop(columns=cols_to_drop, inplace=True)
    
    # Encode Categoricals for Analysis
    for col in analysis_df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        analysis_df[col] = le.fit_transform(analysis_df[col].astype(str))

    columns = analysis_df.columns.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(columns)

    print("   Running Causality Tests...")
    for target in columns:
        for source in columns:
            if source == target: continue
            
            test_data = analysis_df[[target, source]]
            if test_data[source].std() == 0 or test_data[target].std() == 0: continue

            try:
                gc_res = grangercausalitytests(test_data, max_lag, verbose=False)
                min_p_value = 1.0
                is_causal = False
                for lag in range(1, max_lag + 1):
                    p_val = gc_res[lag][0]['ssr_ftest'][1]
                    if p_val < min_p_value: min_p_value = p_val
                    if p_val < p_value_threshold:
                        is_causal = True
                        break 
                
                if is_causal:
                    print(f"      [LINK] {source} -> {target} (p={min_p_value:.4f})")
                    G.add_edge(source, target)
            except: pass

    # --- CRITICAL CHECK ---
    print("\n   [DEBUG] Checking connections for 'boiler_temp':")
    preds = list(G.predecessors('boiler_temp'))
    print(f"      Inputs identified: {preds}")
    if 'operation_mode' not in preds:
        print("      ⚠️ WARNING: 'operation_mode' was NOT linked to 'boiler_temp'. Simulation will fail to react to modes.")
    else:
        print("      ✅ SUCCESS: 'operation_mode' is linked correctly.")

    # Cleanup & Save
    try:
        G_clean = nx.transitive_reduction(G)
        G_clean.add_nodes_from(G.nodes()) 
    except:
        G_clean = G

    if not os.path.exists('plots'): os.makedirs('plots')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G_clean, seed=42, k=2.0)
    nx.draw_networkx_nodes(G_clean, pos, node_size=3000, node_color='lightgreen', edgecolors='black')
    nx.draw_networkx_labels(G_clean, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G_clean, pos, node_size=3000, arrowstyle='-|>', arrowsize=20)
    plt.savefig('plots/structure_graph.png')
    plt.close()
    
    return G_clean

if __name__ == "__main__":
    learn_structure('data/synthetic_plant.csv')