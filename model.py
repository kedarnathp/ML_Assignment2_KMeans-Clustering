
import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers
        self.cluster_centers = X[np.random.choice(X.shape[0], self.num_clusters, replace=False), :]

        for _ in range(max_iter):
            # Assign each sample to the closest prototype
            cluster_assignments = self.predict(X)
            
            # Update prototypes
            new_cluster_centers = []
            for i in range(self.num_clusters):
                cluster_points = X[cluster_assignments == i]
                if cluster_points.size > 0:
                    new_center = cluster_points.mean(axis=0)
                else:
                    new_center = self.cluster_centers[i] 
                new_cluster_centers.append(new_center)

            new_cluster_centers = np.array(new_cluster_centers)

             # Check for convergence
            if np.linalg.norm(new_cluster_centers - self.cluster_centers) < self.epsilon:
                break
            
            self.cluster_centers = new_cluster_centers

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        return np.argmin(distances, axis=1)
        
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        cluster_assignments = self.predict(X)
        return self.cluster_centers[cluster_assignments]
