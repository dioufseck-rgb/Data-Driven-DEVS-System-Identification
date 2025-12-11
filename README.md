AutoTwin: Data-Driven DEVS System Identification
AutoTwin is an automated framework that reverse-engineers simulation models from raw tabular data.
It combines Causal Discovery (to identify system structure), Machine Learning (to learn system physics/logic), and a DEVS-inspired Simulator (to execute the coupled models). This allows you to generate a functioning Digital Twin from a CSV file and run "What-If" scenarios on systems you don't fully understand.
🚀 Key Features
Automated Causal Discovery: Uses Granger Causality to determine which variables drive others (Structure Learning).
Hybrid Data Support: Handles both Numeric (Continuous) and Categorical (Discrete/Logic) data types automatically using generic pipelines.
Modular Learning: Decomposes the system into atomic subsystems and trains specific Random Forest models for each.
DEVS Simulation Engine: A discrete-time execution engine that couples the learned models to simulate future states.
Scenario Injection: Allows users to override specific variables (inputs) to test "What-If" scenarios (e.g., What happens if I turn the system to TURBO mode?).
📂 Project Structure
code
Text
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
🛠️ Installation
Clone the repository (or create the folder).
Install dependencies:
code
Bash
pip install pandas numpy scikit-learn networkx matplotlib statsmodels
🚦 Usage Pipeline
The system works in a 3-stage pipeline: Generate -> Learn -> Simulate.
1. Data Generation (Ground Truth)
Create a synthetic dataset representing a Thermal Power Plant. This plant has a Fuel Valve (Numeric) and an Operation Mode (Categorical: ECO, TURBO, OFF) that affect a Boiler and Turbine.
code
Bash
python src/generator.py
Output: Creates synthetic_plant.csv with frequent mode switching to ensure rich training data.
2. Learning Phase (Structure & Behavior)
This script first runs the Structure Learner to build the dependency graph, then runs the Behavior Learner to train ML pipelines for every identified subsystem.
code
Bash
python src/behavior_learner.py
Output:
Generates plots/structure_graph.png (The Blueprint).
Saves trained models to models/.
Note: Look for "✅ SUCCESS" in the console confirming the link between operation_mode and boiler_temp.
3. Simulation Phase (The Digital Twin)
Run the simulator with a specific "What-If" scenario. The current script simulates a scenario where Fuel is constant, but the Operation Mode switches from ECO 
→
→
 TURBO 
→
→
 OFF.
code
Bash
python src/simulator.py
Output: Generates plots/simulation_hybrid.png.
📊 Understanding the Results
The Structure Graph (plots/structure_graph.png)
The system automatically discovers the physical connections:
fuel_valve 
→
→
 boiler_temp
operation_mode 
→
→
 boiler_temp
boiler_temp 
→
→
 turbine_rpm
The Simulation Plot (plots/simulation_hybrid.png)
The final validation shows the Digital Twin in action:
Green Zone (ECO): System behaves normally.
Yellow Zone (TURBO): The Boiler Temp spikes significantly, even though Fuel Input remained constant. This proves the AI learned the logic of "Turbo Mode."
Gray Zone (OFF): The system cools down immediately, overriding the fuel input.
🧠 How it Works (Under the Hood)
Granger Causality: The system iterates through every pair of columns in the CSV. It tests if the past values of Column A help predict Column B significantly better than Column B's own history.
Graph Construction: Significant links form a Directed Graph. Transitive Reduction is applied to remove "shortcuts" (
A
→
C
A→C
 is removed if 
A
→
B
→
C
A→B→C
 exists).
Hybrid ML Pipeline:
For every node in the graph, we identify its parents (Inputs).
We construct a scikit-learn Pipeline:
Categorical Inputs 
→
→
 OneHotEncoder
Numeric Inputs 
→
→
 Passthrough
Model 
→
→
 RandomForestRegressor
The model learns: 
S
t
a
t
e
t
=
f
(
S
t
a
t
e
t
−
1
,
I
n
p
u
t
s
t
−
1
)
State 
t
​
 =f(State 
t−1
​
 ,Inputs 
t−1
​
 )
DEVS Simulation: The engine initializes the state. At every time step 
t
t
, it:
Accepts external scenario overrides.
Passes current states to connected blocks.
Calculates 
t
+
1
t+1
 for all blocks.
Commits the new state.
