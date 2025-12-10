import pandas as pd
import numpy as np
from scipy import stats
import os


class DataPreprocessor:

    def __init__(self, data_path='data/product_sales.csv'):
        self.df = pd.read_csv(data_path)
        self.missing_summary = None
        self.outlier_summary = {}

    def analyze_missing_values(self):
        # Store number of missing cells
        self.missing_summary = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Data_Type': self.df.dtypes
        })
        return self.missing_summary

    def handle_missing_product_names(self):
        # Sub Data Frame with missing product names
        missing_product_names = self.df[self.df['product_name'].isnull()]

        for index, row in missing_product_names.iterrows():
            # Fetch rows with identical product_id and product_name
            valid_rows = self.df[(self.df['product_id'] == row['product_id']) &
                                (self.df['product_name'].notnull())]

            # Replace missing cell with fetched product_name or default product_name if not found
            if not valid_rows.empty:
                self.df.at[index, 'product_name'] = valid_rows.iloc[0]['product_name']
            else:
                placeholder_name = f"Product_{row['product_id']}"
                self.df.at[index, 'product_name'] = placeholder_name

    def handle_missing_numerical(self, threshold_pct=5):
        # Create data frame with numerical features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Track features that exceed the missing percentage threshold
        damaged_columns = []

        for col in numerical_cols:
            # Count number of missing cells in numerical data frame
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                # Calculate percentage of missing values in column
                missing_pct = (missing_count / len(self.df)) * 100

                if missing_pct < threshold_pct:
                    # If less than threshold is missing impute missing values
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                else:
                    damaged_columns.append(col)

        # Drop rows with missing values in columns that exceed missing threshold
        if damaged_columns:
            self.df.dropna(subset=damaged_columns, inplace=True)

    def detect_outliers_iqr(self, columns=None):
        if columns is None:
            columns = ['price', 'cost', 'units_sold', 'profit', 'promotion_frequency']

        for col in columns:
            # Calculate IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate bounds for outlier detection
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Detect outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.df)) * 100

            # Create summary of outliers for feature
            self.outlier_summary[col] = {
                'count': outlier_count,
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }

        return self.outlier_summary

    def cap_outliers(self, columns=None):
        if columns is None:
            columns = ['price', 'cost', 'units_sold', 'profit', 'promotion_frequency']

        # Create copy of df to cap outliers
        df_no_outliers = self.df.copy()

        for col in columns:
            lower_bound = self.outlier_summary[col]['lower_bound']
            upper_bound = self.outlier_summary[col]['upper_bound']

            # Replace outliers with their respective bounds
            df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)

        return df_no_outliers

    def standardize_zscore(self, df, columns=None):
        if columns is None:
            columns = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level', 'profit']

        df_standardized = df.copy()

        # Z-score normalization
        for col in columns:
            mean_val = df_standardized[col].mean()
            std_val = df_standardized[col].std()
            df_standardized[f'{col}_standardized'] = (df_standardized[col] - mean_val) / std_val

        return df_standardized

    def preprocess_all(self):
        # Analyze missing values
        self.analyze_missing_values()

        # Handle missing values
        self.handle_missing_product_names()
        self.handle_missing_numerical(threshold_pct=5)

        # Detect outliers
        self.detect_outliers_iqr()

        # Cap outliers
        df_capped = self.cap_outliers()

        # Normalize and standardize
        df_zscore = self.standardize_zscore(df_capped)

        return df_capped, df_zscore

    def save_results(self, df_capped, df_zscore, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)

        df_capped.to_csv(f'{output_dir}/preprocessed_data_outliers_capped.csv', index=False)
        df_zscore.to_csv(f'{output_dir}/preprocessed_data_zscore_standardized.csv', index=False)


def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor('data/product_sales.csv')

    # Run all preprocessing steps
    df_capped, df_zscore = preprocessor.preprocess_all()

    # Save results
    preprocessor.save_results(df_capped, df_zscore)


if __name__ == "__main__":
    main()
