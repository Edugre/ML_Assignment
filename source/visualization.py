import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_elbow_curve(k_values, wcss_values, save_path='results/elbow_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    for k, wcss in zip(k_values, wcss_values):
        plt.annotate(f'{wcss:.0f}', (k, wcss), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_clusters_2d(X, labels, centroids, feature_names, save_path='results/cluster_scatter.png'):
    plt.figure(figsize=(12, 8))

    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))

    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

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


def plot_actual_vs_predicted(y_test, predictions, model_name, target, test_r2, test_mse, test_mae,
                             save_path='results/actual_vs_predicted.png'):
    plt.figure(figsize=(10, 8))

    plt.scatter(y_test, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel(f'Actual {target.title()}', fontsize=12)
    plt.ylabel(f'Predicted {target.title()}', fontsize=12)
    plt.title(f'Actual vs Predicted: {model_name}', fontsize=14, fontweight='bold')

    metrics_text = f'R² = {test_r2:.4f}\nMSE = {test_mse:.2f}\nMAE = {test_mae:.2f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(y_test, predictions, model_name, target, save_path='results/residual_plot.png'):
    residuals = y_test - predictions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(predictions, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel(f'Predicted {target.title()}', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(df_comparison, save_path='results/model_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = df_comparison['Model'].values
    x_pos = np.arange(len(models))

    axes[0, 0].bar(x_pos, df_comparison['Test_R2'], color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Test R² Score (Higher is Better)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].bar(x_pos, df_comparison['Test_MSE'], color='coral', edgecolor='black')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Test MSE (Lower is Better)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].bar(x_pos, df_comparison['Test_MAE'], color='lightgreen', edgecolor='black')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Test MAE (Lower is Better)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].bar(x_pos, df_comparison['Overfitting'], color='orange', edgecolor='black')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Train R² - Test R²')
    axes[1, 1].set_title('Overfitting Gap (Lower is Better)')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(feature_names, coefficients, save_path='results/feature_importance.png'):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)

    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in importance_df['Coefficient']]
    plt.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, edgecolor='black')
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance (Linear Regression Coefficients)', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
