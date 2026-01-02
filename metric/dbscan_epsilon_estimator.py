import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def calculate_epsilon(X, k=4, method='knee', sample_size=None):
    """
    Calculate optimal epsilon for DBSCAN clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The data matrix
    k : int, default=4
        Number of nearest neighbors (typically min_samples - 1)
    method : str, default='knee'
        Method to determine epsilon:
        - 'knee': Find the knee/elbow point in k-distance graph
        - 'percentile': Use 90th percentile of k-distances
        - 'manual': Return all k-distances for manual inspection
    sample_size : int or None
        If specified, sample this many points for faster computation
    
    Returns:
    --------
    epsilon : float or array
        Suggested epsilon value(s)
    k_distances : array
        Array of k-nearest neighbor distances (sorted)
    """
    
    # Sample data if dataset is large
    if sample_size is not None and len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X_sample)
    distances, indices = nbrs.kneighbors(X_sample)
    
    # Get k-th nearest neighbor distances (last column)
    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)
    
    if method == 'knee':
        # Find knee point using maximum curvature
        epsilon = find_knee_point(k_distances)
    elif method == 'percentile':
        # Use 90th percentile
        epsilon = np.percentile(k_distances, 90)
    elif method == 'manual':
        epsilon = None
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return epsilon, k_distances


def find_knee_point(k_distances):
    """
    Find the knee/elbow point in the k-distance curve using
    maximum distance from line method.
    """
    n = len(k_distances)
    x = np.arange(n)
    y = k_distances
    
    # Line from first to last point
    p1 = np.array([0, y[0]])
    p2 = np.array([n-1, y[-1]])
    
    # Calculate perpendicular distance of all points to the line
    distances = []
    for i in range(n):
        point = np.array([x[i], y[i]])
        dist = np.abs(np.cross(p2-p1, point-p1)) / np.linalg.norm(p2-p1)
        distances.append(dist)
    
    # Knee is at maximum distance
    knee_idx = np.argmax(distances)
    epsilon = k_distances[knee_idx]
    
    return epsilon


def plot_k_distance_graph(k_distances, epsilon=None, k=4):
    """
    Plot k-distance graph to visualize epsilon selection.
    
    Parameters:
    -----------
    k_distances : array
        Sorted k-nearest neighbor distances
    epsilon : float or None
        If provided, mark this epsilon value on the plot
    k : int
        The k value used
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, linewidth=2)
    
    if epsilon is not None:
        # Find index closest to epsilon
        idx = np.argmin(np.abs(k_distances - epsilon))
        plt.axhline(y=epsilon, color='r', linestyle='--', 
                   label=f'Suggested ε = {epsilon:.4f}')
        plt.plot(idx, epsilon, 'ro', markersize=10)
    
    plt.xlabel('Points sorted by distance', fontsize=12)
    plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
    plt.title(f'K-Distance Graph (k={k})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_blobs
    
    # Create sample dataset
    X, _ = make_blobs(n_samples=1000, n_features=5, centers=3, 
                      cluster_std=1.0, random_state=42)
    
    print("Calculating optimal epsilon for DBSCAN...")
    print("=" * 50)
    
    # Method 1: Knee point detection
    eps_knee, k_dist = calculate_epsilon(X, k=4, method='knee')
    print(f"\nKnee method: ε = {eps_knee:.4f}")
    
    # Method 2: Percentile method
    eps_perc, _ = calculate_epsilon(X, k=4, method='percentile')
    print(f"Percentile method (90th): ε = {eps_perc:.4f}")
    
    # Plot k-distance graph
    print("\nPlotting k-distance graph...")
    plot_k_distance_graph(k_dist, epsilon=eps_knee, k=4)
    
    # For large datasets (like 10M articles), use sampling
    print("\n" + "=" * 50)
    print("For large datasets (e.g., 10M articles):")
    print("Use sample_size parameter:")
    print("eps, k_dist = calculate_epsilon(X, k=4, sample_size=10000)")
