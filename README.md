<<<<<<< HEAD
# VANET Simulation Project

This project simulates a Vehicular Ad-Hoc Network (VANET) using SUMO and Python (TraCI). It features:
- **Vehicle Clustering**: Fitness-guided clustering of vehicles.
- **Inter-Cluster Routing**: Deep Reinforcement Learning (DRLIQ) for routing between Cluster Heads.
- **Intra-Cluster Routing**: Adaptive Q-Learning (AQL) for routing within clusters.
- **Interactive Routing**: Manually trigger routing requests between specific nodes.

## Prerequisites
- **SUMO**: Ensure SUMO is installed and `SUMO_HOME` environment variable is set.
- **Python**: Python 3.x with required libraries (`traci`, `numpy`, `torch`, `scikit-learn`).

## How to Run

### 1. Start the Simulation
Run the main simulation script. This will launch the SUMO GUI and start the simulation logic.

```bash
python run_simulation.py
```

- **Note**: If the SUMO GUI opens but doesn't start automatically, press the **Play** button.
- The simulation logs (Clustering updates, Routing decisions) will appear in the console.

### 2. Interactive Routing (Optional)
To manually trigger a routing request between two vehicles while the simulation is running:

1. Open a **new terminal window**.
2. Navigate to the project directory.
3. Run the message sender script:

```bash
python send_message.py
```

4. Follow the prompts to enter:
   - **Source Node ID** (e.g., `100`)
   - **Destination Node ID** (e.g., `200`)

The simulation will visualize the routing path in the SUMO window:
- **Green Node**: Source
- **Red Node**: Destination
- **Cyan/Orange Arrows**: The computed routing path.

## Project Structure
- `run_simulation.py`: Entry point for the simulation.
- `send_message.py`: Script for sending interactive routing requests.
- `src/simulation/launch_sumo_traci.py`: Main simulation loop and logic.
- `src/ai_core/`: AI models for clustering (K-Means/Fitness) and routing (DRLIQ, AQL).
=======
# vanets_simulation
>>>>>>> 0b161652846cdcee970892aa18adcc3410bb028d
