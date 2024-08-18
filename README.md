# Flanger-DQAF
A data quality assessment platform for ML/AI systems.


## Introduction
Data quality is a critical aspect of any machine learning or AI system. Poor data quality can lead to poor model performance, and in some cases, can even lead to biased or unfair models. In this project, we aim to build a data quality assessment platform for ML/AI systems. The platform will allow users to upload their datasets, and will automatically run a series of data quality checks on the data. The platform will then provide users with a report detailing the results of the data quality checks, as well as recommendations for how to improve the data quality.

</br>

## Installation
To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Getting Started
The sample blueprint is included in the `blueprint.ipynb` file. To run the blueprint, open the file in Jupyter Notebook and run the cells.

The `main.py` file will be updated in future versions to include a command-line toolkit interface for running the data quality assessment platform, as well as data visualization techniques.


## Features and Blueprint

* Data Ingestion and Preprocessing 1.1. Data Format Standardization
* Exploratory Data Analysis (EDA) 2.1. Univariate Analysis
* Statistical Bias Detection 3.1. Sampling Bias Detection (ONLY FOR POPULATION PARAMETERS)
* Systematic Bias Analysis 4.1. Demographic Parity Assessment (SAMPLE POPULATION METRIC)
* Bias Quantification 5.1. Normalized Disparate Impact
* Mitigation Strategies 6.1. Data Reweighting
* Post-Mitigation Validation 7.1. Bias Metric Re-evaluation
* Documentation and Reporting 8.1. Bias Assessment Report


