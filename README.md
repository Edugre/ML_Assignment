# Product Sales ML Analysis

A comprehensive machine learning project for analyzing product sales data using regression models and K-Means clustering. The project includes data preprocessing, multiple regression techniques, custom K-Means implementation, and an interactive Streamlit dashboard for visualization.

## Project Overview

This project analyzes product sales data to:
- Predict profit and units sold using various regression models
- Segment products into clusters using K-Means clustering
- Visualize data insights through an interactive dashboard

## Project Structure

```
Goncalvez_Eduardo_6526311_ML_Assignment/
├── data/
│   └── product_sales.csv                      # Raw sales data
├── source/
│   ├── preprocessing.py                       # Data cleaning and preprocessing
│   ├── kmeans.py                              # Custom K-Means implementation
│   ├── regression.py                          # Regression models
│   ├── visualization.py                       # Plotting functions
│   └── dashboard.py                           # Streamlit dashboard
├── results/                                   # Generated outputs
│   ├── preprocessed_data_outliers_capped.csv
│   ├── preprocessed_data_zscore_standardized.csv
│   ├── kmeans_results.csv
│   ├── regression_results.csv
│   ├── elbow_curve.png
│   ├── clusters_2d.png
│   ├── actual_vs_predicted_Linear.png
│   ├── residuals_Linear.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── requirements.txt                           # Dependencies
├── README.md                                  # This file
└── REPORT.pdf                                 # Project report
```

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start: Run the Dashboard

To view all results and interact with the visualizations:

```bash
streamlit run source/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Complete ML Workflow (Step-by-Step)

If you want to run the entire machine learning pipeline from scratch, execute the files in this order:

### 1. Data Preprocessing
```bash
python source/preprocessing.py
```
**What it does:**
- Loads raw data from `data/product_sales.csv`
- Handles missing values in product names and numerical features
- Detects and caps outliers using IQR method
- Standardizes features using Z-score normalization
- Saves preprocessed data to `results/` directory

**Output files:**
- `results/preprocessed_data_outliers_capped.csv`
- `results/preprocessed_data_zscore_standardized.csv`

### 2. K-Means Clustering
```bash
python source/kmeans.py
```
**What it does:**
- Implements K-Means clustering from scratch (no sklearn)
- Uses K-Means++ initialization for better centroid placement
- Performs elbow method analysis to find optimal number of clusters
- Segments products into clusters based on standardized features

**Output files:**
- `results/kmeans_results.csv`
- `results/elbow_curve.png`
- `results/clusters_2d.png`

### 3. Regression Analysis
```bash
python source/regression.py
```
**What it does:**
- Trains multiple regression models: Linear, Ridge, Lasso, and Polynomial
- Predicts profit based on price, cost, units sold, promotion frequency, and shelf level
- Evaluates models using MSE, MAE, and R² metrics
- Generates comparison visualizations

**Output files:**
- `results/regression_results.csv`
- `results/actual_vs_predicted_Linear.png`
- `results/residuals_Linear.png`
- `results/model_comparison.png`
- `results/feature_importance.png`

### 4. Interactive Dashboard
```bash
streamlit run source/dashboard.py
```
**What it does:**
- Loads all preprocessed data and model results
- Provides interactive visualizations for:
  - Data overview and statistics
  - Missing value analysis
  - Outlier detection
  - Regression model performance
  - K-Means clustering results
- Allows filtering and exploration of data

## Features

### Data Preprocessing
- Missing value imputation using median for numerical features
- Product name inference based on product_id
- Outlier detection using Interquartile Range (IQR)
- Z-score standardization for feature scaling

### Regression Models
- **Linear Regression**: Baseline model
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Lasso Regression**: L1 regularization with feature selection
- **Polynomial Regression**: Captures non-linear relationships

### Clustering
- Custom K-Means implementation with K-Means++ initialization
- Elbow method for optimal cluster selection
- Product segmentation for marketing insights

### Dashboard Features
- Interactive plots using Plotly
- Data filtering and exploration
- Model comparison visualizations
- Cluster analysis and insights
- Downloadable results

## Key Technologies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Regression models and evaluation
- **matplotlib/seaborn**: Static visualizations
- **plotly**: Interactive visualizations
- **streamlit**: Web dashboard
- **scipy**: Statistical analysis

## Requirements

See `requirements.txt` for complete list of dependencies. Key packages:
- Python 3.10+
- pandas >= 2.3.3
- numpy >= 2.2.6
- scikit-learn >= 1.3.0
- streamlit >= 1.28.0
- plotly >= 5.17.0
- matplotlib >= 3.10.7
- seaborn >= 0.13.2

## Results

All generated results (CSV files and visualizations) are saved in the `results/` directory, including:
- Preprocessed datasets
- Model predictions and metrics
- Cluster assignments
- Visualization plots

## Authors

- Eduardo Goncalvez (Student ID: 6526311)
- Alex Waisman (Student ID: 6529880)
- Ivan Salazar (Student ID: 6237206)

## Notes for Professor

- The dashboard (`source/dashboard.py`) provides the quickest way to view all results
- To reproduce the entire analysis, run files 1-3 in order before launching the dashboard
- All visualizations are generated programmatically and saved to the results directory
- The K-Means implementation is custom-built without using sklearn's KMeans class
