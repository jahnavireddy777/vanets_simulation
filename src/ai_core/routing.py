import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Deep Q-Network for DRLIQ
class DRLIQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DRLIQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DRLIQAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DRLIQNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Advanced Q-Learning for Inter-Cluster
class AdvancedQLearningAgent:
    def __init__(self, num_clusters, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((num_clusters, num_clusters)) # State: Current Cluster, Action: Next Cluster
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, neighbors):
        # neighbors is a list of valid next clusters
        if not neighbors:
            return None
        
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(neighbors)
        
        # Select best action among neighbors
        q_values = {n: self.q_table[state, n] for n in neighbors}
        max_q = max(q_values.values())
        # Handle ties
        best_actions = [n for n, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.lr * (target - predict)

if __name__ == "__main__":
    # Test DRLIQ
    agent = DRLIQAgent(state_dim=4, action_dim=2)
    print("DRLIQ Agent initialized.")
    
    # Test Q-Learning
    q_agent = AdvancedQLearningAgent(num_clusters=10)
    print("Q-Learning Agent initialized.")
