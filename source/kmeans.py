import numpy as np
import pandas as pd
import os
from visualization import plot_elbow_curve, plot_clusters_2d

class KMeans:
    def __init__(self, n_clusters = 3, max_iters = 300, random_state = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # WCSS
        self.n_iter_ = 0
    
    def _initialize_centroids_kmeans_plus_plus(self, X):
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
    
    def _assign_clusters(self, X, centroids):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Calculate distances to all centroids
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            # Assign to nearest centroid
            labels[i] = np.argmin(distances)
            
        return labels
    
    def _update_centroids(self, X, labels):
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
    
    def _calculate_wcss(self, X, labels, centroids):
        wcss = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[k])**2)
        return wcss
    
    def _has_converged(self, old_centroids, new_centroids, tol= 1e-4):
        return np.allclose(old_centroids, new_centroids, atol=tol)
    
    def fit(self, X):

        # Initialize centroids
        self.centroids = self._initialize_centroids_kmeans_plus_plus(X)
        
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
    
    def predict(self, X):
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


def elbow_method(X, k_range = None, random_state = 42):

    if k_range is None:
        k_range = list(range(2, 9))  # k = 2 to 8
    
    wcss_values = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)
    
    return k_range, wcss_values

def analyze_clusters(df, labels, centroids, feature_names):

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


def name_and_interpret_clusters(cluster_stats):

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
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
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