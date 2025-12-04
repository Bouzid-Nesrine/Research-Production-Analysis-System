import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

class RIIWeightLearner:
    """
    Learn weights for Research Impact Index (RII) using unsupervised clustering.
    
    RII_i = sum(w_j * x_{i,j}) for all features j
    
    Weights are learned by minimizing intra-cluster variance of RII scores.
    """
    
    def __init__(self, n_clusters=10, learning_rate=0.01, max_iterations=100, 
                 batch_size=10000, random_state=42, non_negative=True):
        """
        Initialize RII weight learner.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for Mini-Batch K-Means
        learning_rate : float
            Learning rate (eta) for gradient descent
        max_iterations : int
            Maximum number of training iterations
        batch_size : int
            Size of mini-batches for training
        random_state : int
            Random seed for reproducibility
        non_negative : bool
            Whether to enforce non-negative weights
        """
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.non_negative = non_negative
        
        self.weights = None
        self.scaler = StandardScaler()
        self.clusterer = MiniBatchKMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            batch_size=batch_size
        )
        self.loss_history = []
        
    def _compute_rii(self, X):
        """Compute RII scores for all samples."""
        return X @ self.weights
    
    def _compute_cluster_variance(self, X, labels):
        """
        Compute total intra-cluster variance of RII scores.
        
        Loss = sum over clusters c of Var_c
        where Var_c = (1/|c|) * sum_{i in c} (RII_i - mean_RII_c)^2
        """
        rii_scores = self._compute_rii(X)
        total_variance = 0.0
        
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if np.sum(mask) == 0:
                continue
                
            cluster_rii = rii_scores[mask]
            cluster_mean = np.mean(cluster_rii)
            cluster_var = np.mean((cluster_rii - cluster_mean) ** 2)
            total_variance += cluster_var
            
        return total_variance
    
    def _compute_gradients(self, X, labels):
        """
        Compute gradient of loss with respect to weights.
        
        ∂Var_c/∂w_j = (2/|c|) * sum_{i in c} (RII_i - mean_RII_c) * (x_{i,j} - mean_x_{c,j})
        """
        rii_scores = self._compute_rii(X)
        n_features = X.shape[1]
        gradients = np.zeros(n_features)
        
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_size = np.sum(mask)
            
            if cluster_size == 0:
                continue
            
            # Get cluster data
            X_cluster = X[mask]
            rii_cluster = rii_scores[mask]
            
            # Compute cluster means
            mean_rii = np.mean(rii_cluster)
            mean_x = np.mean(X_cluster, axis=0)
            
            # Compute gradient for this cluster
            rii_diff = rii_cluster - mean_rii  # shape: (cluster_size,)
            x_diff = X_cluster - mean_x  # shape: (cluster_size, n_features)
            
            # Gradient contribution from this cluster
            cluster_gradient = (2.0 / cluster_size) * (rii_diff[:, np.newaxis] * x_diff).sum(axis=0)
            gradients += cluster_gradient
            
        return gradients
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1."""
        weight_sum = np.sum(np.abs(self.weights))
        if weight_sum > 0:
            self.weights /= weight_sum
            
        if self.non_negative:
            self.weights = np.maximum(self.weights, 0)
            # Re-normalize after clipping
            weight_sum = np.sum(self.weights)
            if weight_sum > 0:
                self.weights /= weight_sum
    
    def fit(self, X, verbose=True):
        """
        Learn RII weights from data using mini-batch training.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix containing bibliometric data
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        n_samples, n_features = X_scaled.shape
        
        # Initialize weights uniformly
        self.weights = np.ones(n_features) / n_features
        
        if verbose:
            print(f"Training RII weight learner on {n_samples} samples with {n_features} features")
            print(f"Using {self.n_clusters} clusters and batch size {self.batch_size}")
        
        # Initial clustering on full dataset (or large sample)
        if n_samples > 100000:
            sample_idx = np.random.choice(n_samples, 100000, replace=False)
            self.clusterer.fit(X_scaled[sample_idx])
        else:
            self.clusterer.fit(X_scaled)
        
        # Mini-batch training
        for iteration in range(self.max_iterations):
            # Sample mini-batch
            batch_idx = np.random.choice(n_samples, 
                                        min(self.batch_size, n_samples), 
                                        replace=False)
            X_batch = X_scaled[batch_idx]
            
            # Get cluster assignments for batch
            labels_batch = self.clusterer.predict(X_batch)
            
            # Compute loss
            loss = self._compute_cluster_variance(X_batch, labels_batch)
            self.loss_history.append(loss)
            
            # Compute gradients
            gradients = self._compute_gradients(X_batch, labels_batch)
            
            # Update weights (gradient descent)
            self.weights -= self.learning_rate * gradients
            
            # Normalize weights
            self._normalize_weights()
            
            # Periodically update clustering
            if iteration % 10 == 0:
                self.clusterer.partial_fit(X_batch)
            
            if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {loss:.6f}")
        
        if verbose:
            print("\nTraining complete!")
            print(f"Final weights: {self.weights}")
            
        return self
    
    def compute_rii(self, X):
        """
        Compute RII scores for new data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        rii_scores : array, shape (n_samples,)
            RII scores for each sample
        """
        X_scaled = self.scaler.transform(X)
        return self._compute_rii(X_scaled)
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on learned weights.
        
        Parameters:
        -----------
        feature_names : list of str, optional
            Names of features
            
        Returns:
        --------
        importance_df : DataFrame
            Feature names and their corresponding weights
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.weights))]
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': self.weights
        })
        importance_df = importance_df.sort_values('Weight', ascending=False)
        
        return importance_df


# Example usage
if __name__ == "__main__":
    # Generate synthetic article data
    np.random.seed(42)
    n_articles = 50000
    
    # Features: citations, num_authors, revenue, years_since_publication, etc.
    data = {
        'citations': np.random.exponential(10, n_articles),
        'num_authors': np.random.poisson(3, n_articles) + 1,
        'revenue': np.random.gamma(2, 1000, n_articles),
        'years_since_pub': np.random.uniform(0, 20, n_articles),
        'h_index_authors': np.random.exponential(5, n_articles)
    }
    
    X = np.column_stack([data[key] for key in data.keys()])
    feature_names = list(data.keys())
    
    # Train the model
    learner = RIIWeightLearner(
        n_clusters=15,
        learning_rate=0.01,
        max_iterations=100,
        batch_size=5000,
        non_negative=True
    )
    
    learner.fit(X, verbose=True)
    
    # Get feature importance
    importance = learner.get_feature_importance(feature_names)
    print("\n" + "="*50)
    print("Feature Importance (Learned Weights):")
    print("="*50)
    print(importance.to_string(index=False))
    
    # Compute RII scores
    rii_scores = learner.compute_rii(X)
    print(f"\nRII Score Statistics:")
    print(f"  Mean: {np.mean(rii_scores):.4f}")
    print(f"  Std:  {np.std(rii_scores):.4f}")
    print(f"  Min:  {np.min(rii_scores):.4f}")
    print(f"  Max:  {np.max(rii_scores):.4f}")
