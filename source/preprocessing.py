import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/product_sales.csv')

missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
    'Data_Type': df.dtypes
})

missing_product_names = df[df['product_name'].isnull()]

for index, row in missing_product_names.iterrows():
    valid_rows = df[(df['product_id'] == row['product_id']) & (df['product_name'].notnull())]

    if not valid_rows.empty:
        df.at[index, 'product_name'] = valid_rows.iloc[0]['product_name']
    else:
        placeholder_name = f"Product_{row['product_id']}"
        df.at[index, 'product_name'] = placeholder_name

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        missing_pct = (missing_count / len(df)) * 100

        if missing_pct < 5:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

outlier_cols = ['price', 'cost', 'units_sold', 'profit', 'promotion_frequency']

outlier_summary = {}

df_no_outliers = df.copy()

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100

    outlier_summary[col] = {
        'count': outlier_count,
        'percentage': outlier_pct,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }

for col in outlier_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers_z = df[z_scores > 3]

for col in outlier_cols:
    lower_bound = outlier_summary[col]['lower_bound']
    upper_bound = outlier_summary[col]['upper_bound']

    original_min = df_no_outliers[col].min()
    original_max = df_no_outliers[col].max()

    df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)

    capped_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

features_to_normalize = ['price', 'cost', 'units_sold', 'promotion_frequency', 'shelf_level', 'profit']

df_minmax = df_no_outliers.copy()

df_zscore = df_no_outliers.copy()

for col in features_to_normalize:
    min_val = df_minmax[col].min()
    max_val = df_minmax[col].max()
    df_minmax[f'{col}_normalized'] = (df_minmax[col] - min_val) / (max_val - min_val)

for col in features_to_normalize:
    mean_val = df_zscore[col].mean()
    std_val = df_zscore[col].std()
    df_zscore[f'{col}_standardized'] = (df_zscore[col] - mean_val) / std_val

df_no_outliers.to_csv('results/preprocessed_data_outliers_capped.csv', index=False)

df_minmax.to_csv('results/preprocessed_data_minmax_normalized.csv', index=False)

df_zscore.to_csv('results/preprocessed_data_zscore_standardized.csv', index=False)
