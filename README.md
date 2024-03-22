# Predict Customer Churn

## Project Description

In this repository, we implement a machine learning-based algorithm to identify credit card customers that are most likely to churn.
The main scope of this repository is to present some software development practices for machine learning such as testing, logging, and best coding practices.

The repository includes Python scripts for the project that follows coding (PEP8) and software engineering best practices for implementing software (modular, documented, and tested). The project can be run from the command-line interface (CLI).

## Running Files

To run the scripts:

**python churn_library.py**

**python test_churn.py**

running the scripts generates the following items:

- EDA plots in the directory ./images/EDA/
- The Machine learning models metrics plots in the directory ./images/metrics/
- The best model pickle files in the directory ./models/
- logs stored in log file ./log/churn_library.log

## Files in the Repo

Below is a tree-like structure of the main directories and files in this repository, along with a brief description of each:

data/
├── bank_data.csv # Contains banking data used for analysis.

images/
├── eda/ # Contains exploratory data analysis images.
│ ├── churn_distribution.png # Distribution of churn across customers.
│ ├── customer_age_distribution.png # Age distribution of customers.
│ ├── heatmap.png # Heatmap of correlations between features.
│ ├── marital_status_distribution.png # Marital status distribution of customers.
│ └── total_transaction_distribution.png # Distribution of total transactions by customers.

├── results/ # Contains images of analysis results.
│ ├── feature_importance.png # Importance of each feature in the model.
│ ├── logistics_results.png # Results from the logistic regression model.
│ ├── rf_results.png # Results from the random forest classifier.
│ └── roc_curve_result.png # ROC curve for the models.

logs/
├── churn_library.log # Log file for the churn library processes.

models/
├── logistic_model.pkl # Pickled logistic regression model.
├── rfc_model.pkl # Pickled random forest classifier model.

## Requirements

Before you can run the programs in this repository, you will need to have the following libraries installed in your Python environment. These libraries are essential for executing the functionalities included in the scripts:

- `numpy`: Used for numerical operations.
- `pandas`: Provides data structures and data analysis tools.
- `matplotlib`: A plotting library for creating static, interactive, and animated visualizations in Python.
- `seaborn`: Based on matplotlib, seaborn provides a high-level interface for drawing attractive and informative statistical graphics.
- `scikit-learn`: Offers simple and efficient tools for predictive data analysis. It is accessible to everybody and reusable in various contexts.
- `joblib`: Used for saving and loading Python objects that make use of NumPy data structures.
- `os`: Provides a way of using operating system dependent functionality.
- `logging`: Used to track events that happen when some software runs.
- `pytest`: A framework that makes it easy to write simple tests, yet scales to support complex functional testing.
