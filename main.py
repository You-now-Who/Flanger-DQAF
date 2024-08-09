import pandas as pd
import matplotlib.pyplot as plt
from bias_reduction import apply_bias_reduction

def load_sample_data():
    """
    This is just a sample dataset I generated via AI LLM models.
    """
    data = {
        'case_duration': [87, 132, 45, 98, 156, 76, 110, 65, 143, 89],
        'defendant_age': [38, 29, 42, 33, 50, 27, 36, 45, 31, 40],
        'judge_experience': [12, 8, 20, 15, 25, 5, 18, 22, 10, 16],
        'case_complexity': ['Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'Low', 'High', 'Medium'],
        'defendant_income': [56234, 78901, 45678, 67890, 89012, 34567, 78901, 56789, 90123, 67890],
        'prior_convictions': [1, 0, 2, 1, 3, 0, 2, 1, 0, 2],
        'legal_representation': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'race': ['White', 'Black', 'Hispanic', 'White', 'Asian', 'White', 'Black', 'Hispanic', 'White', 'Asian'],
        'verdict': ['Guilty', 'Not Guilty', 'Guilty', 'Not Guilty', 'Guilty', 'Not Guilty', 'Guilty', 'Not Guilty', 'Guilty', 'Not Guilty'],
        'case_year': [2018, 2020, 2019, 2021, 2017, 2022, 2018, 2020, 2019, 2021],
        'case_description_length': [200, 350, 150, 250, 400, 180, 300, 220, 380, 270]
    }
    return pd.DataFrame(data)

def analyze_results(df_original, df_processed):
    """
    Perform some basic analysis on the original and processed datasets.
    """
    print("\nOriginal Dataset:")
    print(df_original.describe())
    print("\nProcessed Dataset:")
    print(df_processed.describe())

    # Compare class distribution before and after
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    df_original['verdict'].value_counts().plot(kind='bar')
    plt.title('Original Class Distribution')
    plt.subplot(1, 2, 2)
    df_processed['verdict'].value_counts().plot(kind='bar')
    plt.title('Processed Class Distribution')
    plt.tight_layout()
    plt.show()

def main():
    # Load sample data
    df = load_sample_data()
    print("Sample data loaded:")
    print(df.head())

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
            'label_columns': ['verdict']  # In this sample, we only have one label column
        },
        'popularity_bias': {
            'item_column': 'judge_experience'  # Using judge_experience as a proxy for judge_id
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

    # Analyze results
    analyze_results(df, df_processed)

if __name__ == "__main__":
    main()