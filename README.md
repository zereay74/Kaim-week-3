Here's the fully integrated and updated README that incorporates all your content:

---

# Insurance Data Analysis

A Python framework for processing, analyzing, modeling, and hypothesis testing of insurance datasets. The analysis focuses on understanding trends, relationships, statistical differences, and model-driven insights.

---

## Key Features

- **Data Preparation**: Comprehensive cleaning and transformation of insurance datasets.
- **Feature Engineering**: Create new features like ratios of premiums and claims.
- **Categorical Encoding**: Convert categorical data to numeric formats.
- **Exploratory Analysis**: Visualize trends in premiums, claims, and cover types across time and regions.
- **Statistical Testing**: Perform A/B testing (t-tests, chi-squared tests) to evaluate significant differences in key metrics.
- **Model Building**: Supports Linear Regression, Random Forests, and XGBoost for predictive analysis.
- **Model Evaluation**: Includes MSE, MAE, and R² metrics for assessing performance.
- **Interpretability**: Utilize SHAP values for insights into feature influence on model predictions.

---

## Contents

### Scripts

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

### Notebooks

1. **`Task 1 insurance data.ipynb`**
   - Demonstrates data loading, cleaning, and analysis methods.
   - Examples include:
     - Visualizing premium trends by vehicle make and cover type.
     - Correlation analysis of premiums and claims.
     - Monthly trends in insurance cover types.

2. **`Task 3 statistical testing.ipynb`**
   - Implements statistical hypothesis testing.
   - Examples include:
     - Testing numerical KPIs (e.g., `TotalClaims`) using t-tests.
     - Testing categorical variables (e.g., `MaritalStatus`) using chi-squared tests.
     - Evaluating null hypotheses for insurance risk and margin differences.

3. **`Task 4 model.ipynb`**
   - Interactive analysis and model building.
   - Examples include:
     - Building and tuning predictive models.
     - Evaluating performance metrics.
     - Exploring model interpretability with SHAP.

---

## Getting Started

1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts:
   - Use `data_load_clean_transform.py` to preprocess your dataset.
   - Use `insurance_analysis.py` for exploratory analysis and visualization.
   - Use `hypothesis_testing.py` for A/B testing on segmented data.
   - Use `statistical_modeling.py` to build predictive models.
4. Open the notebooks for guided examples:
   - `Task 1 insurance data.ipynb` for data exploration.
   - `Task 3 statistical testing.ipynb` for hypothesis testing.
   - `Task 4 model.ipynb` for predictive modeling.

---

## Requirements

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost shap
```


