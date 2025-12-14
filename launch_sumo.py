import os
import subprocess

def launch_sumo():
    sumo_gui_path = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
    config_path = os.path.abspath("config/sumo/simulation.sumocfg")
    
    if not os.path.exists(sumo_gui_path):
        print(f"Error: SUMO GUI not found at {sumo_gui_path}")
        return

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Launching SUMO GUI with config: {config_path}")
    subprocess.Popen([sumo_gui_path, "-c", config_path])

if __name__ == "__main__":
    launch_sumo()
