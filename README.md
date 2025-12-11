# AutoTwin: Data-Driven DEVS System Identification

**AutoTwin** is an automated framework that reverse-engineers simulation models from raw tabular data.

It combines **Causal Discovery** (to identify system structure), **Machine Learning** (to learn system physics/logic), and a **DEVS-inspired Simulator** (to execute the coupled models). This allows you to generate a functioning **Digital Twin** from a CSV file and run "What-If" scenarios on systems you don't fully understand.

## 🚀 Key Features

*   **Automated Causal Discovery:** Uses Granger Causality to determine which variables drive others (Structure Learning).
*   **Hybrid Data Support:** Handles both **Numeric** (Continuous) and **Categorical** (Discrete/Logic) data types automatically using generic pipelines.
*   **Modular Learning:** Decomposes the system into atomic subsystems and trains specific Random Forest models for each.
*   **DEVS Simulation Engine:** A discrete-time execution engine that couples the learned models to simulate future states.
*   **Scenario Injection:** Allows users to override specific variables (inputs) to test "What-If" scenarios (e.g., *What happens if I turn the system to TURBO mode?*).

---

## 📂 Project Structure

```text
AutoTwin/
│
├── src/
│   ├── generator.py         # Creates synthetic "Ground Truth" data (Power Plant)
│   ├── structure_learner.py # Step 1: Discovers the wiring (Causal Graph)
│   ├── behavior_learner.py  # Step 2: Learns the math/logic (ML Models)
│   └── simulator.py         # Step 3: Runs the Digital Twin (Execution Engine)
│
├── data/
│   └── synthetic_plant.csv  # Generated input data
│
├── models/                  # Stores trained .pkl models
│   ├── model_boiler_temp.pkl
│   └── model_turbine_rpm.pkl
│
├── plots/                   # Visual outputs
│   ├── structure_graph.png
│   └── simulation_hybrid.png
│
├── README.md                # Documentation
└── requirements.txt         # Dependencies
