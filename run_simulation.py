import sys
import os

# Add the project root to sys.path so we can import from src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.simulation.launch_sumo_traci import run_simulation

if __name__ == "__main__":
    run_simulation()
