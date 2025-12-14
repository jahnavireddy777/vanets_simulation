import random
import math
import numpy as np

class IntraClusterAQL:
    def __init__(self, alpha_base=0.2, gamma_base=0.9, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05):
        self.alpha_base = alpha_base
        self.gamma_base = gamma_base
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-Table: { (current_node, target_node): { neighbor_node: q_value } }
        self.q_table = {}
        
        # Constants
        self.V_MAX = 22.2  # 80 km/h in m/s
        self.RSSI_MIN = -95
        self.RSSI_MAX = -40
        self.BW_MAX = 1.0 # Mbps

    def get_q_value(self, current_node, target_node, action_node):
        state_key = (current_node, target_node)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        return self.q_table[state_key].get(action_node, 0.0)

    def set_q_value(self, current_node, target_node, action_node, value):
        state_key = (current_node, target_node)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_node] = value

    def select_action(self, current_node, target_node, neighbors, neighbor_distances=None):
        """
        Select next hop neighbor using epsilon-greedy strategy.
        neighbors: list of neighbor IDs
        neighbor_distances: dict of {neighbor_id: distance_to_target} (optional heuristic)
        """
        if not neighbors:
            return None
            
        # Exploration
        if random.random() < self.epsilon:
            # Heuristic Fallback during exploration
            if neighbor_distances and random.random() < 0.9:
                 # Filter distances for current neighbors only
                 valid_dists = {n: d for n, d in neighbor_distances.items() if n in neighbors}
                 if valid_dists:
                     return min(valid_dists, key=valid_dists.get)
            return random.choice(neighbors)
            
        # Exploitation
        best_action = None
        max_q = -float('inf')
        
        # Shuffle neighbors to break ties randomly
        random.shuffle(neighbors)
        
        found_trained_q = False
        for neighbor in neighbors:
            q = self.get_q_value(current_node, target_node, neighbor)
            if q != 0.0:
                 found_trained_q = True
            if q > max_q:
                max_q = q
                best_action = neighbor
        
        # If all Q-values are 0 (untrained) and we have heuristics, use heuristic
        if not found_trained_q and neighbor_distances:
            valid_dists = {n: d for n, d in neighbor_distances.items() if n in neighbors}
            if valid_dists:
                 return min(valid_dists, key=valid_dists.get)

        return best_action

    def calculate_adaptive_alpha(self, v_self, v_neighbor):
        """
        Alpha = alpha_base * (1 + beta * normalized_relative_speed)
        beta = 1.0
        """
        relative_speed = abs(v_self - v_neighbor)
        normalized_rel_speed = relative_speed / self.V_MAX
        alpha = self.alpha_base * (1 + 1.0 * normalized_rel_speed)
        return max(0.0, min(1.0, alpha))

    def calculate_adaptive_gamma(self, rssi, bandwidth):
        """
        Gamma = gamma_base * (w_rssi * RSSI_norm + w_bw * BW_norm)
        w_rssi = 0.6, w_bw = 0.4
        """
        rssi_norm = (rssi - self.RSSI_MIN) / (self.RSSI_MAX - self.RSSI_MIN)
        rssi_norm = max(0.0, min(1.0, rssi_norm))
        
        bw_norm = bandwidth / self.BW_MAX
        bw_norm = max(0.0, min(1.0, bw_norm))
        
        gamma = self.gamma_base * (0.6 * rssi_norm + 0.4 * bw_norm)
        return max(0.0, min(1.0, gamma))

    def calculate_reward(self, success, old_dist, new_dist, failure_type=None):
        """
        Calculate reward based on outcome.
        """
        if success:
            return 10.0
        
        if failure_type == 'link_failure':
            return -15.0
        elif failure_type == 'backward':
            return -5.0
        elif failure_type == 'loop':
            return -20.0
            
        # Progress Reward
        progress = old_dist - new_dist
        return 0.5 * progress

    def update(self, current_node, target_node, action_node, reward, next_node, next_neighbors, 
               v_self, v_neighbor, rssi, bandwidth):
        """
        Perform Q-Learning update.
        """
        # Calculate adaptive parameters
        alpha = self.calculate_adaptive_alpha(v_self, v_neighbor)
        gamma = self.calculate_adaptive_gamma(rssi, bandwidth)
        
        # Get current Q
        old_q = self.get_q_value(current_node, target_node, action_node)
        
        # Calculate max Q for next state
        max_q_next = 0.0
        if next_neighbors:
            q_values = [self.get_q_value(next_node, target_node, n) for n in next_neighbors]
            max_q_next = max(q_values)
            
        # Bellman Update
        # Q_new = Q_old + alpha * (reward + gamma * max_Q_next - Q_old)
        new_q = old_q + alpha * (reward + gamma * max_q_next - old_q)
        
        self.set_q_value(current_node, target_node, action_node, new_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
