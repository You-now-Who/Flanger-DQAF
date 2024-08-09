import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def reduce_class_imbalance(df, target_column, categorical_columns=None):
    """
    Reduce class imbalance using SMOTE.
    
    :param df: Input DataFrame
    :param target_column: Name of the target column
    :param categorical_columns: List of categorical columns to encode
    :return: Resampled DataFrame
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return pd.concat([X_resampled, y_resampled], axis=1)

def reduce_feature_bias(df, target_column, n_features=5):
    """
    Reduce feature bias by selecting top k features.
    
    :param df: Input DataFrame
    :param target_column: Name of the target column
    :param n_features: Number of top features to select
    :return: DataFrame with selected features
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return df[selected_features + [target_column]], selected_features

def reduce_temporal_bias(df, year_column, value_column, base_year=None):
    """
    Adjust a value column for inflation based on the year.
    
    :param df: Input DataFrame
    :param year_column: Name of the year column
    :param value_column: Name of the value column to adjust
    :param base_year: Base year for adjustment (default: minimum year in the dataset)
    :return: DataFrame with adjusted value column
    """
    if base_year is None:
        base_year = df[year_column].min()
    
    df['adjusted_value'] = df[value_column] * (1 + 0.02) ** (df[year_column] - base_year)
    return df

def reduce_label_bias(df, label_columns):
    """
    Reduce label bias by taking the mode of multiple label columns.
    
    :param df: Input DataFrame
    :param label_columns: List of columns containing labels
    :return: DataFrame with a new 'final_label' column
    """
    df['final_label'] = df[label_columns].mode(axis=1)[0]
    return df

def reduce_popularity_bias(df, item_column):
    """
    Reduce popularity bias by assigning weights inversely proportional to item frequency.
    
    :param df: Input DataFrame
    :param item_column: Name of the column containing items (e.g., judge_id)
    :return: DataFrame with a new weight column
    """
    item_counts = df[item_column].value_counts()
    df['popularity_weight'] = 1 / df[item_column].map(item_counts)
    return df

def reduce_length_bias(df, duration_column, length_column):
    """
    Reduce length bias by normalizing a duration column by a length column.
    
    :param df: Input DataFrame
    :param duration_column: Name of the duration column
    :param length_column: Name of the length column
    :return: DataFrame with a new normalized duration column
    """
    df['normalized_duration'] = df[duration_column] / df[length_column]
    return df

def reduce_outlier_bias(df, columns_to_check, contamination=0.1):
    """
    Identify outliers using Isolation Forest.
    
    :param df: Input DataFrame
    :param columns_to_check: List of columns to check for outliers
    :param contamination: The proportion of outliers in the dataset
    :return: DataFrame with a new 'is_outlier' column
    """
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = isolation_forest.fit_predict(df[columns_to_check])
    df['is_outlier'] = outliers
    return df

def apply_bias_reduction(df, config):
    """
    Apply multiple bias reduction techniques based on the provided configuration.
    
    :param df: Input DataFrame
    :param config: Dictionary containing configuration for each bias reduction technique
    :return: Processed DataFrame
    """
    if 'class_imbalance' in config:
        df = reduce_class_imbalance(df, config['class_imbalance']['target_column'], 
                                    config['class_imbalance'].get('categorical_columns'))
    
    if 'feature_bias' in config:
        df, selected_features = reduce_feature_bias(df, config['feature_bias']['target_column'], 
                                                    config['feature_bias'].get('n_features', 5))
        print(f"Selected features: {selected_features}")
    
    if 'temporal_bias' in config:
        df = reduce_temporal_bias(df, config['temporal_bias']['year_column'], 
                                  config['temporal_bias']['value_column'], 
                                  config['temporal_bias'].get('base_year'))
    
    if 'label_bias' in config:
        df = reduce_label_bias(df, config['label_bias']['label_columns'])
    
    if 'popularity_bias' in config:
        df = reduce_popularity_bias(df, config['popularity_bias']['item_column'])
    
    if 'length_bias' in config:
        df = reduce_length_bias(df, config['length_bias']['duration_column'], 
                                config['length_bias']['length_column'])
    
    if 'outlier_bias' in config:
        df = reduce_outlier_bias(df, config['outlier_bias']['columns_to_check'], 
                                 config['outlier_bias'].get('contamination', 0.1))
    
    return df

# Example usage
if __name__ == "__main__":
    # Load your CSV file
    df = pd.read_csv("your_dataset.csv")
    
    # Define configuration for bias reduction
    config = {
        'class_imbalance': {
            'target_column': 'verdict',
            'categorical_columns': ['case_complexity', 'legal_representation', 'gender', 'race']
        },
        'feature_bias': {
            'target_column': 'verdict',
            'n_features': 5
        },
        'temporal_bias': {
            'year_column': 'case_year',
            'value_column': 'defendant_income'
        },
        'label_bias': {
            'label_columns': ['verdict', 'verdict_annotator1', 'verdict_annotator2']
        },
        'popularity_bias': {
            'item_column': 'judge_id'
        },
        'length_bias': {
            'duration_column': 'case_duration',
            'length_column': 'case_description_length'
        },
        'outlier_bias': {
            'columns_to_check': ['case_duration', 'defendant_age', 'judge_experience', 'defendant_income'],
            'contamination': 0.1
        }
    }
    
    # Apply bias reduction techniques
    df_processed = apply_bias_reduction(df, config)
    
    # Display results
    print(df_processed.head())
    print(df_processed.describe())
