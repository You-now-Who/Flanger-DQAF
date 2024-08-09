"""
A program to assess the quality of a given dataset.
Quality refers to bias, fairness, and accuracy of the dataset.
This is aimed at datasets that are used for training machine learning models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the dataset
def load_dataset(file_path):
    """
    Load the dataset from a given file path.
    """
    dataset = pd.read_csv(file_path)
    return dataset

def main():
    # Load the dataset
    dataset = load_dataset("data/german_credit_data.csv")
    print(dataset.head())

    # Check for missing values
    missing_values = dataset.isnull().sum()
    print(missing_values)

    # Check for duplicates
    duplicates = dataset.duplicated().sum()
    print(duplicates)

    # Check for bias
    bias = dataset["Risk"].value_counts()
    print(bias)

    # Check