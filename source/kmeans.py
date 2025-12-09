import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import os

class KMeans:
    def __init__(self, n_clusters: int = 3, max_iters: int = 300, random_state: int = 42, init_method: str = 'kmeans++'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.init_method = init_method
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # WCSS
        self.n_iter_ = 0
        
    def _initialize_centroids_random(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices].copy()
    
    def _initialize_centroids_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        centroids = [X[np.random.randint(n_samples)]]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Calculate distances from each point to nearest centroid
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for idx, prob in enumerate(cumulative_probs):
                if r < prob:
                    centroids.append(X[idx])
                    break
                    
        return np.array(centroids)
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Calculate distances to all centroids
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            # Assign to nearest centroid
            labels[i] = np.argmin(distances)
            
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Get all points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Update centroid as mean of cluster points
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[k] = X[np.random.randint(X.shape[0])]
                
        return new_centroids
    
    def _calculate_wcss(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        wcss = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[k])**2)
        return wcss
    
    def _has_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray, tol: float = 1e-4) -> bool:
        return np.allclose(old_centroids, new_centroids, atol=tol)
    
    def fit(self, X: np.ndarray):

        # Initialize centroids
        if self.init_method == 'kmeans++':
            self.centroids = self._initialize_centroids_kmeans_plus_plus(X)
        else:
            self.centroids = self._initialize_centroids_random(X)
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iters):
            # Step 2: Assign points to nearest centroid
            self.labels_ = self._assign_clusters(X, self.centroids)
            
            # Step 3: Update centroids
            new_centroids = self._update_centroids(X, self.labels_)
            
            # Step 4: Check for convergence
            if self._has_converged(self.centroids, new_centroids):
                self.centroids = new_centroids
                self.n_iter_ = iteration + 1
                break
                
            self.centroids = new_centroids
        else:
            self.n_iter_ = self.max_iters
        
        # Calculate final WCSS
        self.inertia_ = self._calculate_wcss(X, self.labels_, self.centroids)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


def elbow_method(X: np.ndarray, k_range: List[int] = None, random_state: int = 42) -> Tuple[List[int], List[float]]:

    if k_range is None:
        k_range = list(range(2, 9))  # k = 2 to 8
    
    wcss_values = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, init_method='kmeans++')
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)
    
    return k_range, wcss_values


def plot_elbow_curve(k_values: List[int], wcss_values: List[float], save_path: str = 'results/elbow_curve.png'):

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Add value labels on points
    for k, wcss in zip(k_values, wcss_values):
        plt.annotate(f'{wcss:.0f}', (k, wcss), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray, 
                     feature_names: List[str]) -> pd.DataFrame:

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    cluster_stats = []
    
    for cluster_id in range(len(centroids)):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        stats = {
            'Cluster': cluster_id,
            'Count': len(cluster_data),
            'Avg_Price': cluster_data['price'].mean(),
            'Avg_Units_Sold': cluster_data['units_sold'].mean(),
            'Avg_Profit': cluster_data['profit'].mean(),
            'Avg_Promotion_Freq': cluster_data['promotion_frequency'].mean(),
            'Avg_Cost': cluster_data['cost'].mean(),
            'Total_Profit': cluster_data['profit'].sum(),
            'Total_Units_Sold': cluster_data['units_sold'].sum()
        }
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)


def name_and_interpret_clusters(cluster_stats: pd.DataFrame) -> pd.DataFrame:

    cluster_names = []
    insights = []
    
    for idx, row in cluster_stats.iterrows():
        # Classify based on price and volume
        price_level = "Budget" if row['Avg_Price'] < 3 else "Mid-Range" if row['Avg_Price'] < 6 else "Premium"
        volume_level = "High-Volume" if row['Avg_Units_Sold'] > 600 else "Steady" if row['Avg_Units_Sold'] > 300 else "Low-Volume"
        
        # Create descriptive name
        if row['Avg_Units_Sold'] > 600 and row['Avg_Price'] < 3:
            name = "Budget Best-Sellers"
            insight = "High-volume, low-price products driving revenue through quantity. Focus on maintaining stock levels and supply chain efficiency."
        elif row['Avg_Price'] > 6 and row['Avg_Units_Sold'] < 300:
            name = "Premium Specialty"
            insight = "High-margin, low-volume premium products. Consider targeted marketing and ensure quality positioning."
        elif row['Avg_Price'] >= 3 and row['Avg_Price'] <= 6 and row['Avg_Units_Sold'] >= 300:
            name = "Mid-Range Steady Performers"
            insight = "Reliable products with balanced pricing and consistent sales. Maintain current strategy and monitor competition."
        elif row['Avg_Promotion_Freq'] > 2:
            name = "Promotion-Driven Products"
            insight = "Products requiring frequent promotions to drive sales. Evaluate pricing strategy and product positioning."
        else:
            name = f"{price_level} {volume_level}"
            insight = f"Products with {price_level.lower()} pricing and {volume_level.lower()} sales pattern."
        
        cluster_names.append(name)
        insights.append(insight)
    
    cluster_stats['Cluster_Name'] = cluster_names
    cluster_stats['Business_Insight'] = insights
    
    return cluster_stats


def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, 
                     feature_names: List[str], save_path: str = 'results/cluster_scatter.png'):

    plt.figure(figsize=(12, 8))
    
    # Use first two features (typically price and units_sold after preprocessing)
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))
    
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', marker='X', s=500, edgecolors='black', linewidth=2,
               label='Centroids', zorder=5)
    
    plt.xlabel(f'{feature_names[0]} (standardized)', fontsize=12)
    plt.ylabel(f'{feature_names[1]} (standardized)', fontsize=12)
    plt.title('K-means Clustering: Product Segmentation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    
    # Load preprocessed data
    df = pd.read_csv('results/preprocessed_data_zscore_standardized.csv')
    
    # Select features for clustering
    feature_cols = ['price_standardized', 'units_sold_standardized', 
                   'profit_standardized', 'promotion_frequency_standardized']
    X = df[feature_cols].values
    
    # Elbow Method
    k_range = list(range(2, 9))
    k_values, wcss_values = elbow_method(X, k_range, random_state=42)
    
    # Plot elbow curve
    plot_elbow_curve(k_values, wcss_values)
    
    # Determine optimal k
    optimal_k = 4
    
    # Run K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, init_method='kmeans++')
    labels = kmeans.fit_predict(X)
    
    # Analyze clusters
    cluster_stats = analyze_clusters(df, labels, kmeans.centroids, feature_cols)
    cluster_stats = name_and_interpret_clusters(cluster_stats)
    
    # Save cluster statistics
    cluster_stats.to_csv('results/cluster_analysis.csv', index=False)
    
    # Create visualizations
    plot_clusters_2d(X[:, :2], labels, kmeans.centroids[:, :2], 
                    ['Price', 'Units Sold'])
    
    # Save labeled data
    df_final = df.copy()
    df_final['cluster'] = labels
    df_final.to_csv('results/data_with_clusters.csv', index=False)


if __name__ == "__main__":
    main()