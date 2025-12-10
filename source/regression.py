import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


class RegressionAnalysis:

    def __init__(self, data_path='results/preprocessed_data_outliers_capped.csv'):
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def prepare_data(self, target='profit', test_size=0.3, random_state=42):

        # Select features for prediction
        # Exclude target and non-predictive columns
        exclude_cols = ['product_id', 'product_name', 'category', target]
        
        if target == 'profit':
            # When predicting profit, we can use: price, cost, units_sold, promotion_frequency, shelf_level
            feature_cols = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level']
        else:  # predicting units_sold
            # When predicting units_sold: price, cost, promotion_frequency, shelf_level, profit
            feature_cols = ['price', 'cost', 'promotion_frequency', 'shelf_level', 'profit']
        
        self.feature_names = feature_cols
        X = self.df[feature_cols].values
        y = self.df[target].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.target = target
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'model': model,
            'name': 'Linear Regression',
            'train_mse': mean_squared_error(self.y_train, y_pred_train),
            'test_mse': mean_squared_error(self.y_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'predictions': y_pred_test,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
        
        self.models['linear'] = metrics
        return metrics
    
    def train_polynomial_regression(self, degree=2):

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train)
        X_test_poly = poly.transform(self.X_test)
        
        # Train linear regression on polynomial features
        model = LinearRegression()
        model.fit(X_train_poly, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        
        # Calculate metrics
        metrics = {
            'model': model,
            'name': f'Polynomial Regression (degree={degree})',
            'degree': degree,
            'poly_features': poly,
            'train_mse': mean_squared_error(self.y_train, y_pred_train),
            'test_mse': mean_squared_error(self.y_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'predictions': y_pred_test,
            'n_features': X_train_poly.shape[1]
        }
        
        self.models[f'poly_{degree}'] = metrics
        return metrics
    
    def train_ridge_regression(self, alpha=1.0):

        model = Ridge(alpha=alpha)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'model': model,
            'name': f'Ridge Regression (alpha={alpha})',
            'train_mse': mean_squared_error(self.y_train, y_pred_train),
            'test_mse': mean_squared_error(self.y_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'predictions': y_pred_test,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
        
        self.models['ridge'] = metrics
        return metrics
    
    def compare_models(self):
        comparison = []
        
        for key, metrics in self.models.items():
            comparison.append({
                'Model': metrics['name'],
                'Train_MSE': metrics['train_mse'],
                'Test_MSE': metrics['test_mse'],
                'Train_MAE': metrics['train_mae'],
                'Test_MAE': metrics['test_mae'],
                'Train_R2': metrics['train_r2'],
                'Test_R2': metrics['test_r2'],
                'Overfitting': metrics['train_r2'] - metrics['test_r2']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('Test_R2', ascending=False)
        
        return df_comparison
    
    def get_best_model(self):
        best_key = max(self.models.keys(), 
                      key=lambda k: self.models[k]['test_r2'])
        return best_key, self.models[best_key]
    
    def plot_actual_vs_predicted(self, model_key=None, save_path='results/actual_vs_predicted.png'):

        if model_key is None:
            model_key, model_data = self.get_best_model()
        else:
            model_data = self.models[model_key]
        
        predictions = model_data['predictions']
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(self.y_test, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line (diagonal)
        min_val = min(self.y_test.min(), predictions.min())
        max_val = max(self.y_test.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel(f'Actual {self.target.title()}', fontsize=12)
        plt.ylabel(f'Predicted {self.target.title()}', fontsize=12)
        plt.title(f'Actual vs Predicted: {model_data["name"]}', fontsize=14, fontweight='bold')
        
        # Add metrics as text
        r2 = model_data['test_r2']
        mse = model_data['test_mse']
        mae = model_data['test_mae']
        
        metrics_text = f'R² = {r2:.4f}\nMSE = {mse:.2f}\nMAE = {mae:.2f}'
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(self, model_key=None, save_path='results/residual_plot.png'):

        if model_key is None:
            model_key, model_data = self.get_best_model()
        else:
            model_data = self.models[model_key]
        
        predictions = model_data['predictions']
        residuals = self.y_test - predictions
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual scatter plot
        ax1.scatter(predictions, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel(f'Predicted {self.target.title()}', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residual histogram
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_data["name"]}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, save_path='results/model_comparison.png'):
        """Plot comparison of all models."""
        df_comparison = self.compare_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = df_comparison['Model'].values
        x_pos = np.arange(len(models))
        
        # Test R2
        axes[0, 0].bar(x_pos, df_comparison['Test_R2'], color='steelblue', edgecolor='black')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Test R² Score (Higher is Better)')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Test MSE
        axes[0, 1].bar(x_pos, df_comparison['Test_MSE'], color='coral', edgecolor='black')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('Test MSE (Lower is Better)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Test MAE
        axes[1, 0].bar(x_pos, df_comparison['Test_MAE'], color='lightgreen', edgecolor='black')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Test MAE (Lower is Better)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Overfitting gap
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
    
    def plot_feature_importance(self, save_path='results/feature_importance.png'):

        if 'linear' not in self.models:
            return
        
        model_data = self.models['linear']
        coefficients = model_data['coefficients']
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
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


def main():
    
    # Initialize analysis
    analysis = RegressionAnalysis()
    
    # Prepare data (predicting profit)
    analysis.prepare_data(target='profit', test_size=0.3, random_state=42)
    
    # Train models
    analysis.train_linear_regression()
    analysis.train_polynomial_regression(degree=2)
    analysis.train_polynomial_regression(degree=3)
    analysis.train_ridge_regression(alpha=1.0)
    
    # Compare models
    df_comparison = analysis.compare_models()
    df_comparison.to_csv('results/regression_comparison.csv', index=False)
    
    # Get best model
    best_key, best_model = analysis.get_best_model()
    
    # Save detailed results
    with open('results/regression_analysis.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("REGRESSION ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Target Variable: {analysis.target}\n")
        f.write(f"Features Used: {', '.join(analysis.feature_names)}\n")
        f.write(f"Training Set Size: {len(analysis.y_train)}\n")
        f.write(f"Test Set Size: {len(analysis.y_test)}\n\n")
        
        f.write("="*60 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("="*60 + "\n\n")
        f.write(df_comparison.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*60 + "\n")
        f.write(f"BEST MODEL: {best_model['name']}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test R² Score: {best_model['test_r2']:.4f}\n")
        f.write(f"Test MSE: {best_model['test_mse']:.2f}\n")
        f.write(f"Test MAE: {best_model['test_mae']:.2f}\n\n")
        
        f.write("INTERPRETATION:\n")
        
        # Determine best model and explain
        if best_model['test_r2'] > 0.9:
            performance = "excellent"
        elif best_model['test_r2'] > 0.7:
            performance = "good"
        elif best_model['test_r2'] > 0.5:
            performance = "moderate"
        else:
            performance = "poor"
        
        f.write(f"- The {best_model['name']} shows {performance} performance (R²={best_model['test_r2']:.4f})\n")
        
        # Check for overfitting
        overfitting_gap = best_model['train_r2'] - best_model['test_r2']
        if overfitting_gap > 0.1:
            f.write(f"- WARNING: Signs of overfitting detected (gap={overfitting_gap:.4f})\n")
        else:
            f.write(f"- No significant overfitting detected (gap={overfitting_gap:.4f})\n")
        
        # Compare linear vs polynomial
        if 'poly_2' in analysis.models:
            linear_r2 = analysis.models['linear']['test_r2']
            poly_r2 = analysis.models['poly_2']['test_r2']
            if poly_r2 > linear_r2:
                f.write(f"- Polynomial features improve performance by {(poly_r2-linear_r2)*100:.2f}%\n")
            else:
                f.write(f"- Linear model performs better, relationships are mostly linear\n")
    
    # Create visualizations
    analysis.plot_actual_vs_predicted()
    analysis.plot_residuals()
    analysis.plot_model_comparison()
    analysis.plot_feature_importance()


if __name__ == "__main__":
    main()