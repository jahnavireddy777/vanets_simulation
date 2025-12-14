import os
import sys
import traci
import random
import time
import math
import numpy as np
import json
from sklearn.neighbors import KDTree

# Ensure SUMO_HOME is set
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# Add project root to sys.path to allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our clustering logic
# Assuming run_simulation.py sets up the path correctly, we can import from src.ai_core
from src.ai_core.clustering import VehicleClustering
from src.ai_core.drliq import DRLIQAgent, StateCalculator
from src.ai_core.aql import IntraClusterAQL

def create_circle_shape(center, radius, num_points=32):
    shape = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        shape.append((x, y))
    return shape

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("simulation.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_simulation():
    sys.stdout = Logger()
    sumo_binary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
    # Config path relative to this script: ../../config/sumo/simulation.sumocfg
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(base_dir, "../../config/sumo/simulation.sumocfg"))
    
    # Start SUMO with TraCI
    sumo_cmd = [sumo_binary, "-c", config_path, "--start"]
    traci.start(sumo_cmd)
    
    step = 0
    accident_caused = False
    accident_vehicle = None
    
    # Clustering setup
    clustering_interval = 50 # Run clustering every 50 steps
    n_clusters = 25 # Increased for better coverage
    cluster_algo = VehicleClustering(n_clusters=n_clusters)
    
    # DRLIQ Initialization
    state_calc = StateCalculator()
    # State: 4 metrics. Action: Select one of n_clusters to forward to
    drliq_agent = DRLIQAgent(state_dim=4, action_dim=n_clusters)
    
    # AQL Initialization (Intra-Cluster)
    aql_agent = IntraClusterAQL()
    
    cluster_colors = [
        (255, 0, 0, 255),   # Red
        (0, 255, 0, 255),   # Green
        (0, 0, 255, 255),   # Blue
        (255, 255, 0, 255), # Yellow
        (255, 0, 255, 255), # Magenta
        (0, 255, 255, 255), # Cyan
        (128, 0, 128, 255), # Purple
        (255, 165, 0, 255)  # Orange
    ]

    # Vehicle State Tracking (for RE simulation)
    vehicle_states = {} # {vid: {'RE': 100.0}}

    # Initialize clustering variables to ensure they exist before first clustering step
    labels = None
    centers = None
    ch_indices_map = {}
    valid_vehicles = [] # List of VIDs from last clustering
    positions_np = None # Positions from last clustering (still useful for cluster ID lookup if we assume clusters don't move too fast, or we could re-calc)

    print("Simulation started. Press Play in SUMO if not auto-started.")
    
    while step < 1000:
        traci.simulationStep()
        
        # --- Data Collection & Simulation ---
        current_vehicles = traci.vehicle.getIDList()
        
        # Update/Initialize Vehicle States
        for vid in current_vehicles:
            if vid not in vehicle_states:
                vehicle_states[vid] = {'RE': 100.0} # Start with 100% energy
            else:
                # Simulate energy decay (0.01% to 0.05% per step)
                decay = random.uniform(0.01, 0.05)
                vehicle_states[vid]['RE'] = max(0.0, vehicle_states[vid]['RE'] - decay)

        # Remove vehicles that left
        vehicle_states = {k: v for k, v in vehicle_states.items() if k in current_vehicles}

        # --- Clustering Logic ---
        if step % clustering_interval == 0 and step > 0:
            if len(current_vehicles) > n_clusters:
                positions = []
                current_valid_vehicles = [] # Temp list
                re_list = []
                jitter_list = []
                
                for vid in current_vehicles:
                    try:
                        pos = traci.vehicle.getPosition(vid)
                        positions.append(pos)
                        current_valid_vehicles.append(vid)
                        re_list.append(vehicle_states[vid]['RE'])
                        # Simulate Jitter (0-10ms)
                        jitter_list.append(random.uniform(1, 10))
                    except:
                        pass
                
                if positions:
                    positions_np = np.array(positions)
                    valid_vehicles = current_valid_vehicles # Update the global list
                    
                    # Calculate Neighbor Degree (ND)
                    # Radius for neighbors = 250m (typical WiFi range)
                    tree = KDTree(positions_np)
                    # query_radius returns count of neighbors within radius
                    nd_list = tree.query_radius(positions_np, r=250, count_only=True)
                    
                    # Calculate Link Quality Indicator (LQI)
                    lqi_list = [nd * random.uniform(0.8, 1.0) for nd in nd_list]

                    # Prepare data for clustering
                    vehicle_data = {
                        'RE': np.array(re_list),
                        'Jitter': np.array(jitter_list),
                        'ND': np.array(nd_list),
                        'LQI': np.array(lqi_list)
                    }

                    # Perform Fitness-Guided Clustering
                    labels = cluster_algo.fit(positions_np, vehicle_data)
                    centers = cluster_algo.cluster_centers_
                    ch_indices_map = cluster_algo.get_cluster_heads(positions_np)
                    
                    # Log Results
                    print(f"\n--- Step {step}: Clustering Update ---")
                    print(f"Total Vehicles: {len(valid_vehicles)}")
                    
                    # --- Export Active Vehicles for External Tools ---
                    try:
                        active_veh_file = os.path.join(base_dir, "../../data/active_vehicles.json")
                        with open(active_veh_file, "w") as f:
                            json.dump({
                                "timestamp": time.time(),
                                "step": step,
                                "vehicles": sorted(valid_vehicles)
                            }, f)
                    except Exception as e:
                        print(f"Error exporting active vehicles: {e}")
                    
                    # Remove old polygons
                    for poly in traci.polygon.getIDList():
                        if poly.startswith("cluster_") or poly.startswith("ch_") or poly.startswith("aql_arrow_") or poly.startswith("user_req_") or poly.startswith("path_"):
                            traci.polygon.remove(poly)
                    
                    for poi in traci.poi.getIDList():
                        if poi.startswith("label_"):
                            traci.poi.remove(poi)

                    def add_label(vid, pos):
                        poi_id = str(vid)
                        try:
                            # Remove if exists (to update pos)
                            traci.poi.remove(poi_id)
                        except:
                            pass
                        try:
                            traci.poi.add(poi_id, pos[0], pos[1], (255, 255, 255, 255), "text", layer=100, height=0)
                        except:
                            pass

                    # Clean up numeric POIs from previous step
                    for poi in traci.poi.getIDList():
                        if poi.isdigit():
                            traci.poi.remove(poi)

                    # Draw new clusters and CHs
                    for i in range(n_clusters):
                        # Find vehicles in this cluster
                        cluster_indices = np.where(labels == i)[0]
                        if len(cluster_indices) == 0:
                            continue
                            
                        # Cluster Head
                        ch_idx = ch_indices_map.get(i)
                        ch_vid = None
                        if ch_idx is not None:
                            ch_vid = valid_vehicles[ch_idx]
                            ch_pos = positions_np[ch_idx]
                            ch_fitness = cluster_algo.fitness_scores_[ch_idx]
                            
                            print(f"Cluster {i}: CH={ch_vid}, Fitness={ch_fitness:.2f}, Size={len(cluster_indices)}")
                            
                            # Highlight CH by coloring the vehicle white
                            try:
                                traci.vehicle.setColor(ch_vid, (255, 255, 255, 255)) # White for CH
                            except:
                                pass

                        # Draw Cluster Boundary
                        cluster_points = positions_np[cluster_indices]
                        center = centers[i]
                        distances = np.linalg.norm(cluster_points - center, axis=1)
                        # Even tighter: 60th percentile, no buffer
                        radius = np.percentile(distances, 60)
                        
                        shape = create_circle_shape(center, radius)
                        color = cluster_colors[i % len(cluster_colors)]
                        # Thinner line width (2 instead of 3)
                        try:
                            traci.polygon.add(f"cluster_{i}", shape, color, fill=False, layer=10, lineWidth=2)
                        except:
                            pass
                        
                        # Color Member Vehicles
                        veh_color = (color[0], color[1], color[2], 255)
                        for idx in cluster_indices:
                            vid = valid_vehicles[idx]
                            if vid != accident_vehicle and vid != ch_vid: # Don't overwrite accident or CH
                                try:
                                    traci.vehicle.setColor(vid, veh_color)
                                except:
                                    pass

        # --- Accident Logic ---
        if step == 100 and not accident_caused:
            vehicles = traci.vehicle.getIDList()
            if vehicles:
                accident_vehicle = vehicles[0]
                accident_lane = traci.vehicle.getLaneID(accident_vehicle)
                try:
                    traci.vehicle.setStop(accident_vehicle, accident_lane, pos=50, duration=200)
                    traci.vehicle.setColor(accident_vehicle, (255, 0, 0, 255)) # Red for accident
                    print(f"ACCIDENT SIMULATED: Vehicle {accident_vehicle} stopped.")
                    accident_caused = True
                except:
                    pass

        # --- Interactive Routing Request Handling ---
        request_file = os.path.join(base_dir, "../../data/routing_request.json")
        if step % 100 == 0:
            print(f"Checking for request at: {request_file}")
        
        if os.path.exists(request_file):
            try:
                with open(request_file, "r") as f:
                    req_data = json.load(f)
                
                src_node = req_data.get("source_node")
                dst_node = req_data.get("destination_node")
                
                print(f"\n--- Received User Routing Request ---")
                print(f"Source: {src_node} -> Destination: {dst_node}")
                
                # Validation
                if src_node in current_vehicles and dst_node in current_vehicles:
                    src_pos = traci.vehicle.getPosition(src_node)
                    dst_pos = traci.vehicle.getPosition(dst_node)
                    
                    # Visualize Request
                    try:
                        traci.polygon.add("user_req_line", [src_pos, dst_pos], (255, 255, 255, 255), fill=False, layer=30, lineWidth=1)
                        traci.vehicle.setColor(src_node, (0, 255, 0, 255)) # Green Source
                        traci.vehicle.setColor(dst_node, (255, 0, 0, 255)) # Red Dest
                    except:
                        pass
                    
                    # Helper to get realtime neighbors
                    def get_realtime_neighbors(vid, radius=250):
                        try:
                            pos = traci.vehicle.getPosition(vid)
                            pos_np = np.array(pos)
                            neighbors = []
                            for other in current_vehicles:
                                if other == vid: continue
                                try:
                                    other_pos = np.array(traci.vehicle.getPosition(other))
                                    dist = np.linalg.norm(pos_np - other_pos)
                                    if dist <= radius:
                                        neighbors.append(other)
                                except:
                                    pass
                            return neighbors
                        except:
                            return []

                    # Determine Clusters
                    def get_cluster_id(pos, centers):
                        if centers is None: return -1
                        dists = np.linalg.norm(centers - pos, axis=1)
                        return np.argmin(dists)
                    
                    # Check if we have clustering data
                    if centers is not None:
                        c_src = get_cluster_id(src_pos, centers)
                        c_dst = get_cluster_id(dst_pos, centers)
                        
                        print(f"Source Cluster: {c_src}, Destination Cluster: {c_dst}")
                        
                        # Find CH VIDs for these clusters (using the map from last clustering step)
                        # The CH VID is stable between clustering intervals, but its POSITION must be updated.
                        src_ch_vid = None
                        dst_ch_vid = None
                        
                        if c_src in ch_indices_map and ch_indices_map[c_src] < len(valid_vehicles):
                             src_ch_vid = valid_vehicles[ch_indices_map[c_src]]
                             
                        if c_dst in ch_indices_map and ch_indices_map[c_dst] < len(valid_vehicles):
                             dst_ch_vid = valid_vehicles[ch_indices_map[c_dst]]

                        # Check if Source and Dest ARE the CHs themselves
                        is_src_ch = (src_node == src_ch_vid)
                        is_dst_ch = (dst_node == dst_ch_vid)
                        
                        # Add labels for Source and Destination
                        add_label(src_node, src_pos)
                        add_label(dst_node, dst_pos)

                        # --- Path Tracing Helper Functions ---
                        def trace_aql_path(start_vid, target_vid, agent, neighbor_radius=250, max_hops=15):
                            path_points = []
                            path_vids = []
                            
                            curr_vid = start_vid
                            try:
                                curr_pos = traci.vehicle.getPosition(curr_vid)
                                path_points.append(curr_pos)
                                path_vids.append(curr_vid)
                            except:
                                return [], []

                            try:
                                target_pos = np.array(traci.vehicle.getPosition(target_vid))
                            except:
                                return [], [] # Target gone

                            for _ in range(max_hops):
                                if curr_vid == target_vid:
                                    break
                                
                                # Get neighbors
                                neighbors = get_realtime_neighbors(curr_vid, radius=neighbor_radius)
                                # Filter out visited to avoid loops
                                neighbors = [n for n in neighbors if n not in path_vids]
                                
                                # Heuristic distances
                                neighbor_dists = {}
                                for n in neighbors:
                                    try:
                                        n_pos = np.array(traci.vehicle.getPosition(n))
                                        neighbor_dists[n] = np.linalg.norm(n_pos - target_pos)
                                    except:
                                        pass
                                
                                # Select Next Hop
                                next_hop = agent.select_action(curr_vid, target_vid, neighbors, neighbor_distances=neighbor_dists)
                                
                                if next_hop:
                                    curr_vid = next_hop
                                    try:
                                        curr_pos = traci.vehicle.getPosition(curr_vid)
                                        path_points.append(curr_pos)
                                        path_vids.append(curr_vid)
                                    except:
                                        break
                                else:
                                    # Dead end
                                    break
                            
                            return path_points, path_vids

                        def trace_drliq_path(start_ch_vid, target_ch_vid, agent, ch_map, valid_vehs, max_hops=10):
                            path_points = []
                            path_vids = []
                            curr_vid = start_ch_vid
                            
                            try:
                                curr_pos = traci.vehicle.getPosition(curr_vid)
                                path_points.append(curr_pos)
                                path_vids.append(curr_vid)
                                target_pos = np.array(traci.vehicle.getPosition(target_ch_vid))
                            except:
                                return [], []

                            # Map vid -> cluster_idx
                            vid_to_cluster = {v: k for k,v in ch_map.items() if v < len(valid_vehs) and valid_vehs[v] == v} # imperfect check
                            # Better: rebuild reverse map from current ch_indices_map
                            cluster_idx_to_vid = {}
                            for c, v_idx in ch_map.items():
                                if v_idx < len(valid_vehs):
                                    cluster_idx_to_vid[c] = valid_vehs[v_idx]

                            for _ in range(max_hops):
                                if curr_vid == target_ch_vid:
                                    break
                                
                                curr_pos_np = np.array(path_points[-1])
                                
                                # Find potential next CHs (physically reachable, e.g. 500m for CH-CH logic check)
                                potential_next_clusters = []
                                valid_actions = []
                                
                                for c_idx, ch_vid in cluster_idx_to_vid.items():
                                    if ch_vid == curr_vid or ch_vid in path_vids:
                                        continue
                                    try:
                                        ch_pos = np.array(traci.vehicle.getPosition(ch_vid))
                                        dist = np.linalg.norm(curr_pos_np - ch_pos)
                                        # Looser constraint for CH selection, but we will fill gap with AQL
                                        if dist <= 600: 
                                            potential_next_clusters.append(c_idx)
                                            valid_actions.append(c_idx)
                                    except:
                                        pass
                                
                                if not valid_actions:
                                    break

                                # Prepare State
                                dist_to_dst = np.linalg.norm(curr_pos_np - target_pos)
                                state = state_calc.get_state(dist_to_dst, random.uniform(0, 30))
                                
                                # Heuristic
                                heuristic_action = None
                                min_dist = float('inf')
                                for c_idx in valid_actions:
                                    ch_vid = cluster_idx_to_vid[c_idx]
                                    try:
                                        p = np.array(traci.vehicle.getPosition(ch_vid))
                                        d = np.linalg.norm(p - target_pos)
                                        if d < min_dist:
                                            min_dist = d
                                            heuristic_action = c_idx
                                    except:
                                        pass

                                action_c_idx = agent.select_action(state, valid_actions=valid_actions, heuristic_action=heuristic_action)
                                
                                if action_c_idx in cluster_idx_to_vid:
                                    next_ch_vid = cluster_idx_to_vid[action_c_idx]
                                    
                                    # Now, route FROM curr_vid TO next_ch_vid using AQL (Multi-hop fill)
                                    print(f"  DRLIQ Hop: {curr_vid} -> {next_ch_vid}. Finding intermediate nodes...")
                                    sub_points, sub_vids = trace_aql_path(curr_vid, next_ch_vid, IntraClusterAQL(), max_hops=10) # Use new AQL instance or existing
                                    
                                    if sub_points:
                                        # We successfully walked to the next CH
                                        # Append all points except the very first one (which matches path_points[-1]) if duplicate
                                        if path_points and sub_points and np.array_equal(path_points[-1], sub_points[0]):
                                             path_points.extend(sub_points[1:])
                                             path_vids.extend(sub_vids[1:])
                                        else:
                                             path_points.extend(sub_points)
                                             path_vids.extend(sub_vids)
                                        
                                        curr_vid = next_ch_vid
                                    else:
                                        # Gap is too big for AQL (vehicles too sparse), but DRLIQ selected it.
                                        # Visual Fallback: Draw a dashed line or just jump (this is the "teleport" case we want to avoid, but might be necessary if density is low)
                                        print(f"  Warning: No physical path to next CH {next_ch_vid}. Jump forced.")
                                        try:
                                            ch_pos_real = traci.vehicle.getPosition(next_ch_vid)
                                            path_points.append(ch_pos_real)
                                            path_vids.append(next_ch_vid)
                                            curr_vid = next_ch_vid
                                        except:
                                            break

                                else:
                                    break
                                    
                            return path_points, path_vids


                        # --- Execution & Visualization ---
                        print(f"Routing logic for {src_node} -> {dst_node}")
                        
                        full_path_points = []
                        
                        # Case A: Same Cluster (Intra)
                        if c_src == c_dst and c_src != -1:
                            print("Strategy: Intra-Cluster (AQL Full Path)")
                            points, vids = trace_aql_path(src_node, dst_node, aql_agent)
                            if points:
                                full_path_points.extend(points)
                                print(f"Path Found: {vids}")
                            else:
                                print("Path finding failed (Intra).")

                        # Case B: Different Cluster (Hybrid)
                        else:
                            print("Strategy: Inter-Cluster (Hybrid: AQL -> DRLIQ -> AQL)")
                            
                            # 1. Src -> SrcCH
                            if src_ch_vid:
                                points1, vids1 = trace_aql_path(src_node, src_ch_vid, aql_agent)
                                if points1:
                                    full_path_points.extend(points1)
                                    print(f"Segment 1 (to CH): {vids1}")
                                else:
                                    print("Segment 1 failed.")
                            
                                # 2. SrcCH -> DstCH
                                if src_ch_vid and dst_ch_vid:
                                    points2, vids2 = trace_drliq_path(src_ch_vid, dst_ch_vid, drliq_agent, ch_indices_map, valid_vehicles)
                                    if points2:
                                        # Avoid duplicating the join point
                                        if full_path_points:
                                            full_path_points.extend(points2[1:])
                                        else:
                                            full_path_points.extend(points2)
                                        print(f"Segment 2 (CH-CH): {vids2}")
                                    else:
                                        print("Segment 2 failed (no CH path).")
                                    
                                    # 3. DstCH -> Dst
                                    points3, vids3 = trace_aql_path(dst_ch_vid, dst_node, aql_agent)
                                    if points3:
                                        if full_path_points:
                                            full_path_points.extend(points3[1:])
                                        else:
                                            full_path_points.extend(points3)
                                        print(f"Segment 3 (to Dst): {vids3}")
                                    else:
                                        print("Segment 3 failed.")
                                else:
                                     print("Cannot route: Missing CHs.")
                            else:
                                # Fallback: Try direct AQL if no CH logic works
                                print("Fallback: Direct AQL global search")
                                points, vids = trace_aql_path(src_node, dst_node, aql_agent, max_hops=30)
                                full_path_points = points

                        # Visualize Final Path
                        if len(full_path_points) > 1:
                            try:
                                # Draw thick polyline
                                traci.polygon.add("path_final", full_path_points, (0, 255, 255, 255), fill=False, layer=40, lineWidth=4)
                                
                                # Add hop markers
                                for idx, p in enumerate(full_path_points):
                                    traci.poi.add(f"hop_{idx}", p[0], p[1], (255, 165, 0, 255), "ok", layer=41, width=1, height=1)
                                    
                                print("Path visualization updated.")
                            except Exception as e:
                                print(f"Vis Error: {e}")
                        else:
                            print("No complete path found to visualize.")
                    else:
                        print("Clustering not yet initialized. Wait for step 50.")
                        
                else:
                    print("Error: Source or Destination node not found in simulation.")
                
                # Remove file to process only once
                os.remove(request_file)
                
            except Exception as e:
                print(f"Error processing request: {e}")
                # Print stack trace
                import traceback
                traceback.print_exc()

        step += 1

    traci.close()

if __name__ == "__main__":
    run_simulation()
