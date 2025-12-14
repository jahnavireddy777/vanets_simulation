import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from scipy.special import erfc

# --- State Calculator ---
class StateCalculator:
    def __init__(self):
        # Constants for formulas
        self.B = 1e6  # Bandwidth (1 Mbps)
        self.P_tx = 20  # Transmit Power (dBm) -> 100mW
        self.G_t = 1.0  # Antenna Gain Tx
        self.G_r = 1.0  # Antenna Gain Rx
        self.freq = 5.9e9  # Frequency (5.9 GHz)
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / self.freq
        self.N0_dBm = -174  # Noise power density (dBm/Hz)
        self.N0 = 10**((self.N0_dBm - 30) / 10) # Watts/Hz
        
        # Load parameters
        self.alpha = 0.3
        self.beta = 0.5
        self.gamma = 0.8
        self.theta = 0.1
        self.M_i = 1e6 # Max capacity (bytes/sec)

    def calculate_P_ij(self, dist, relative_velocity, time_window=1.0):
        """
        Communication Interruption Probability (P_ij)
        P_ij = 1 - exp(-lambda * T)
        Lambda depends on velocity and distance. Simplified model:
        lambda = relative_velocity / (CommunicationRange - dist + epsilon)
        """
        comm_range = 250.0 # meters
        if dist >= comm_range:
            return 1.0 # Link broken
        
        # Simplified link breakage rate
        # As dist approaches range, lambda increases
        # As velocity increases, lambda increases
        epsilon = 1e-6
        lambda_val = relative_velocity / (comm_range - dist + epsilon)
        
        P_ij = 1 - math.exp(-lambda_val * time_window)
        return max(0.0, min(1.0, P_ij))

    def calculate_L_ij(self, data_to_send, hist_avg, queue_len, data_gen_rate, recv_rate):
        """
        Vehicle Node Load (L_ij)
        L_ij = [Sum(d + alpha*h) + beta(q + r)] / [M - gamma*exp(-theta*o)]
        """
        numerator = (data_to_send + self.alpha * hist_avg) + self.beta * (data_gen_rate + recv_rate)
        denominator = self.M_i - self.gamma * math.exp(-self.theta * queue_len)
        
        if denominator <= 0:
            return 1.0 # Saturated
            
        L_ij = numerator / denominator
        return max(0.0, min(1.0, L_ij)) # Normalize to 0-1

    def calculate_R_ij(self, dist):
        """
        Transmission Rate (R'_ij)
        Shannon-Hartley: B * log2(1 + SNR)
        SNR = Pr / Noise
        Pr = (Pt * Gt * Gr * lambda^2) / (4 * pi * d)^2 * Loss
        """
        if dist <= 0: dist = 0.1
        
        # Free space path loss component
        path_loss_fs = (self.wavelength / (4 * math.pi * dist)) ** 2
        
        # Additional losses (Multipath, Obstacles) - Simulated
        L_multi = 1.5
        L_obs = 1.0 # Assuming LOS for CH-CH often
        total_loss_factor = 1 / (L_multi * L_obs)
        
        # Transmit power in Watts
        P_tx_watts = 10**((self.P_tx - 30) / 10)
        
        # Received power
        Pr = P_tx_watts * self.G_t * self.G_r * path_loss_fs * total_loss_factor
        
        # Noise Power (Bandwidth * N0)
        Noise = self.B * self.N0
        
        SNR = Pr / Noise
        
        R_ij = self.B * math.log2(1 + SNR)
        
        # Normalize R_ij (e.g., relative to max bandwidth)
        return min(1.0, R_ij / (10 * self.B)) # Cap at 1.0 for state

    def calculate_Pe_ij(self, dist):
        """
        Transmission Error Rate (Pe'_ij)
        Pe = 0.5 * erfc(sqrt(SNR))
        """
        # Re-calculate SNR (reuse logic or simplify)
        if dist <= 0: dist = 0.1
        path_loss_fs = (self.wavelength / (4 * math.pi * dist)) ** 2
        P_tx_watts = 10**((self.P_tx - 30) / 10)
        Pr = P_tx_watts * path_loss_fs # Simplified
        Noise = self.B * self.N0
        SNR = Pr / Noise
        
        Pe = 0.5 * erfc(math.sqrt(SNR))
        return Pe

    def get_state(self, dist, rel_vel, queue_len=0):
        """
        Returns normalized state vector [P_ij, L_ij, R_ij, Pe_ij]
        Simulating some parameters for now.
        """
        P = self.calculate_P_ij(dist, rel_vel)
        L = self.calculate_L_ij(1000, 500, queue_len, 100, 100) # Mock load params
        R = self.calculate_R_ij(dist)
        Pe = self.calculate_Pe_ij(dist)
        
        return np.array([P, L, R, Pe], dtype=np.float32)

# --- DRL Agent ---
class DRLIQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DRLIQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class DRLIQAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DRLIQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DRLIQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state, valid_actions=None, heuristic_action=None):
        """
        Select action (next hop CH index).
        Select action (next hop CH index).
        valid_actions: list of valid CH indices (neighbors)
        heuristic_action: index of the best known action based on heuristics (e.g. invalid Q-values)
        """
        if np.random.rand() <= self.epsilon:
            if heuristic_action is not None and random.random() < 0.9:
                 return heuristic_action
            if valid_actions:
                return random.choice(valid_actions)
            return random.randrange(self.action_dim)
            
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
            
        # Mask invalid actions
        if valid_actions is not None:
            # Create a mask of -inf for invalid actions
            full_q = q_values.cpu().numpy()[0]
            masked_q = np.full_like(full_q, -np.inf)
            masked_q[valid_actions] = full_q[valid_actions]
            return np.argmax(masked_q)
            
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)
        
        # Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # V(s') = max Q(s', a') from target net
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self, tau=0.005):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def calculate_reward(self, state):
        """
        Reward based on target metrics.
        state: [P, L, R, Pe]
        Targets: P=0.1, L=0.5, R=0.8, Pe=0.01
        """
        P, L, R, Pe = state
        
        # Targets
        P_target = 0.1
        L_target = 0.5
        R_target = 0.8
        Pe_target = 0.01
        
        reward = (math.exp(-abs(P - P_target)) + 
                  math.exp(-abs(L - L_target)) + 
                  math.exp(-abs(R - R_target)) + 
                  math.exp(-abs(Pe - Pe_target)))
                  
        return reward
