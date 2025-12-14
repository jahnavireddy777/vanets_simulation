import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

class VehicleClustering:
    def __init__(self, n_clusters=5, batch_size=100, weights=(0.4, 0.2, 0.2, 0.2)):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.weights = weights # w1(RE), w2(Jitter), w3(ND), w4(LQI)
        self.kmeans = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.fitness_scores_ = None

    def calculate_fitness(self, vehicle_data):
        """
        Calculate fitness score for each vehicle.
        vehicle_data: dict containing arrays for 'RE', 'Jitter', 'ND', 'LQI'
        Fitness(i) = w1 * RE + w2 * (1/Jitter) + w3 * ND + w4 * LQI
        """
        re = vehicle_data['RE']
        jitter = vehicle_data['Jitter']
        nd = vehicle_data['ND']
        lqi = vehicle_data['LQI']

        # Avoid division by zero for jitter
        jitter_inv = 1.0 / (jitter + 1e-6)

        # Normalize components to 0-1 range to make weights meaningful
        # (Simple min-max normalization)
        def normalize(x):
            if np.max(x) == np.min(x): return np.ones_like(x)
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        re_norm = normalize(re)
        jitter_norm = normalize(jitter_inv)
        nd_norm = normalize(nd)
        lqi_norm = normalize(lqi)

        w1, w2, w3, w4 = self.weights
        fitness = (w1 * re_norm) + (w2 * jitter_norm) + (w3 * nd_norm) + (w4 * lqi_norm)
        return fitness

    def fit(self, vehicle_positions, vehicle_data):
        """
        Fit the clustering model using Fitness-Guided initialization.
        vehicle_positions: np.array (n_vehicles, 2)
        vehicle_data: dict with keys 'RE', 'Jitter', 'ND', 'LQI'
        """
        n_samples = vehicle_positions.shape[0]
        if n_samples < self.n_clusters:
            # Fallback if not enough vehicles
            self.n_clusters = max(1, n_samples)
        
        # 1. Calculate Fitness
        self.fitness_scores_ = self.calculate_fitness(vehicle_data)

        # 2. Intelligent Centroid Initialization
        # Select top-k vehicles with highest fitness as initial centroids
        # We need to ensure centroids are somewhat spread out, but strictly following 
        # the prompt: "top-k vehicles with the highest fitness scores are selected"
        top_k_indices = np.argsort(self.fitness_scores_)[-self.n_clusters:]
        initial_centroids = vehicle_positions[top_k_indices]

        # 3. MiniBatchKMeans with custom init
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            init=initial_centroids,
            n_init=1, # Explicit init provided
            max_iter=10, # Limited iterations as per prompt
            random_state=42
        )
        
        self.kmeans.fit(vehicle_positions)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = self.kmeans.labels_
        
        return self.labels_

    def get_cluster_heads(self, vehicle_positions):
        """
        Select Cluster Head for each cluster based on highest fitness.
        Returns: dict {cluster_id: vehicle_index_in_input_array}
        """
        ch_indices = {}
        for i in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Get fitness scores for vehicles in this cluster
            cluster_fitness = self.fitness_scores_[cluster_indices]
            
            # Find index of max fitness within this cluster
            max_fitness_idx_local = np.argmax(cluster_fitness)
            max_fitness_idx_global = cluster_indices[max_fitness_idx_local]
            
            ch_indices[i] = max_fitness_idx_global
            
        return ch_indices
