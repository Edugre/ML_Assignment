import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from visualization import plot_actual_vs_predicted, plot_residuals, plot_model_comparison, plot_feature_importance


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

        plot_actual_vs_predicted(
            self.y_test,
            predictions,
            model_data['name'],
            self.target,
            model_data['test_r2'],
            model_data['test_mse'],
            model_data['test_mae'],
            save_path
        )
    
    def plot_residuals(self, model_key=None, save_path='results/residual_plot.png'):
        if model_key is None:
            model_key, model_data = self.get_best_model()
        else:
            model_data = self.models[model_key]

        predictions = model_data['predictions']

        plot_residuals(
            self.y_test,
            predictions,
            model_data['name'],
            self.target,
            save_path
        )
    
    def plot_model_comparison(self, save_path='results/model_comparison.png'):
        df_comparison = self.compare_models()
        plot_model_comparison(df_comparison, save_path)
    
    def plot_feature_importance(self, save_path='results/feature_importance.png'):
        if 'linear' not in self.models:
            return

        model_data = self.models['linear']
        coefficients = model_data['coefficients']

        plot_feature_importance(self.feature_names, coefficients, save_path)

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
    
    # Create visualizations
    analysis.plot_actual_vs_predicted()
    analysis.plot_residuals()
    analysis.plot_model_comparison()
    analysis.plot_feature_importance()


if __name__ == "__main__":
    main()