import json
import os
import time

def send_message():
    print("--- VANET Interactive Routing ---")
    print("Enter the details for the routing request.")
    
    try:
        # Load active vehicles if available
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        active_veh_file = os.path.join(data_dir, "active_vehicles.json")
        
        active_vehicles = []
        if os.path.exists(active_veh_file):
            try:
                with open(active_veh_file, "r") as f:
                    data = json.load(f)
                    active_vehicles = data.get("vehicles", [])
                    print(f"\nActive Vehicles ({len(active_vehicles)}): {active_vehicles[:10]} ...")
            except:
                pass

        src_node = input("Source Node ID (e.g., 10): ").strip()
        # src_cluster = input("Source Cluster ID (Optional, press Enter to skip): ").strip()
        
        if active_vehicles and src_node not in active_vehicles:
            print(f"Warning: Source Node '{src_node}' is not currently active in the simulation.")
        
        dst_node = input("Destination Node ID (e.g., 25): ").strip()
        # dst_cluster = input("Destination Cluster ID (Optional, press Enter to skip): ").strip()

        if active_vehicles and dst_node not in active_vehicles:
            print(f"Warning: Destination Node '{dst_node}' is not currently active in the simulation.")
        
        if not src_node or not dst_node:
            print("Error: Source and Destination Node IDs are required.")
            return

        request_data = {
            "source_node": src_node,
            "destination_node": dst_node,
            "timestamp": time.time()
        }
        
        # Path to the data directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        request_file = os.path.join(data_dir, "routing_request.json")
        
        with open(request_file, "w") as f:
            json.dump(request_data, f, indent=4)
            
        print(f"\nRequest sent! Check the simulation window/console.")
        print(f"Request written to: {request_file}")
        
    except KeyboardInterrupt:
        print("\nCancelled.")

if __name__ == "__main__":
    while True:
        send_message()
        again = input("\nSend another message? (y/n): ").strip().lower()
        if again != 'y':
            break
