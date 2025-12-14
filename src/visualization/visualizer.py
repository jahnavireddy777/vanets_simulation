import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.ai_core.clustering import VehicleClustering
from src.ai_core.routing import DRLIQAgent, AdvancedQLearningAgent

class VanetVisualizer:
    def __init__(self, n_vehicles=100, area_size=1000):
        self.n_vehicles = n_vehicles
        self.area_size = area_size
        self.vehicles = np.random.rand(n_vehicles, 2) * area_size
        self.energies = np.random.rand(n_vehicles)
        
        # Initialize AI modules
        self.clustering = VehicleClustering(n_clusters=int(n_vehicles/10))
        self.drliq = DRLIQAgent(state_dim=4, action_dim=2)
        self.q_learning = AdvancedQLearningAgent(num_clusters=int(n_vehicles/10))
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.scat = self.ax.scatter(self.vehicles[:, 0], self.vehicles[:, 1], c='b', s=50)
        self.cluster_centers_plot = self.ax.scatter([], [], c='r', s=200, marker='x')
        self.lines = []

    def update(self, frame):
        # 1. Move vehicles (random walk for demo)
        self.vehicles += np.random.randn(self.n_vehicles, 2) * 5
        self.vehicles = np.clip(self.vehicles, 0, self.area_size)
        
        # 2. Perform Clustering
        labels = self.clustering.fit(self.vehicles)
        centers = self.clustering.cluster_centers_
        
        # 3. Update Visualization
        self.scat.set_offsets(self.vehicles)
        self.scat.set_array(labels) # Color by cluster
        self.cluster_centers_plot.set_offsets(centers)
        
        # 4. Draw Routing Paths (Mockup)
        # Clear old lines
        for line in self.lines:
            line.remove()
        self.lines = []
        
        # Draw lines from vehicles to their cluster center
        for i, vehicle in enumerate(self.vehicles):
            center = centers[labels[i]]
            line, = self.ax.plot([vehicle[0], center[0]], [vehicle[1], center[1]], 'k-', alpha=0.1)
            self.lines.append(line)
            
        self.ax.set_title(f"VANET Simulation - Frame {frame}")
        return self.scat, self.cluster_centers_plot, *self.lines

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=200, interval=100, blit=False)
        plt.show()

if __name__ == "__main__":
    viz = VanetVisualizer()
    viz.run()
