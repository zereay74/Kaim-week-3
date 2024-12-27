# Insurance Data Analysis

This repository contains scripts and notebooks for analyzing an insurance dataset. The analysis focuses on understanding trends, relationships, and distributions of key variables such as premiums, claims, and cover types.

## Contents

### Scripts
1. **`data_load_clean_transform.py`**
   - Purpose: Load, clean, and transform the insurance dataset.
   - Key Features:
     - Handles missing values and outliers.
     - Converts date and time columns to a standardized timezone.
     - Prepares the dataset for analysis.

2. **`insurance_analysis.py`**
   - Purpose: Perform data analysis and visualization on the insurance dataset.
   - Key Features:
     - Identifies numerical and categorical columns.
     - Conducts univariate, bivariate, and multivariate analyses.
     - Generates visualizations for trends over time and geography.
     - Explores correlations between premiums and claims by category.

### Notebook
- **`Task 1 insurance data.ipynb`**
  - Demonstrates the implementation of the data loading, cleaning, and analysis methods.
  - Includes examples of:
    - Visualizing premium trends by vehicle make and cover type.
    - Correlation analysis of premiums and claims.
    - Monthly trends in insurance cover types.

## Getting Started
1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts:
   - Use `data_load_clean_transform.py` to preprocess your dataset.
   - Use `insurance_analysis.py` to perform detailed analysis and generate plots.
4. Open the `Task 1 insurance data.ipynb` notebook to view step-by-step examples.

## Highlights
- Comprehensive cleaning and transformation of insurance datasets.
- Detailed exploratory data analysis with clear and insightful visualizations.
- Modular scripts for flexibility and reuse in similar projects.



