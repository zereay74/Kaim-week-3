# Scripts

1. **`data_load_clean_transform.py`**
   - **Purpose**: Load, clean, and transform the insurance dataset.
   - **Key Features**:
     - Handles missing values and outliers.
     - Converts date and time columns to a standardized timezone.
     - Prepares the dataset for analysis.

2. **`insurance_analysis.py`**
   - **Purpose**: Perform data analysis and visualization on the insurance dataset.
   - **Key Features**:
     - Identifies numerical and categorical columns.
     - Conducts univariate, bivariate, and multivariate analyses.
     - Generates visualizations for trends over time and geography.
     - Explores correlations between premiums and claims by category.

3. **`hypothesis_testing.py`**
   - **Purpose**: Conduct A/B hypothesis testing on segmented insurance data.
   - **Key Features**:
     - Supports numerical testing (t-tests) and categorical testing (chi-squared).
     - Segments data into control and test groups based on features like province or zip code.
     - Evaluates null hypotheses for risk, margin, and demographic differences.

4. **`statistical_modeling.py`**
   - **Purpose**: Build and evaluate predictive models using insurance data.
   - **Key Features**:
     - Supports Linear Regression, Random Forest, and XGBoost.
     - Includes evaluation metrics (MSE, MAE, R²).
     - Implements SHAP for interpretability.

---