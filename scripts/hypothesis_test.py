import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

class ABHypothesisTesting:
    def __init__(self, data):
        """
        Initialize the A/B Hypothesis Testing class.

        :param data: Pandas DataFrame 
        """
        self.data = data
        self.group_a = None
        self.group_b = None
        self.kpi = None  # Initialize KPI attribute

    def select_kpi(self, kpi_column):
        """
        Select the Key Performance Indicator (KPI) column.

        :param kpi_column: Column name for the KPI.
        """
        self.kpi = kpi_column
        print(f"Selected KPI: {self.kpi}")

    def segment_data(self, column_name, group_a_value, group_b_value):
        if column_name not in self.data.columns:
            raise ValueError(f"Segmentation column '{column_name}' not found in the dataset.")
        
        # Check for missing or inconsistent values
        self.data[column_name] = self.data[column_name].str.strip().str.title()  # Clean strings

        # Check if the group values exist in the column
        unique_values = self.data[column_name].unique()
        if group_a_value not in unique_values or group_b_value not in unique_values:
            raise ValueError(
                f"One or both of the segmentation values '{group_a_value}' and '{group_b_value}' are not in the column. "
                f"Available values: {unique_values}"
            )
        
        # Segment the data into Group A and Group B
        self.group_a = self.data[self.data[column_name] == group_a_value]
        self.group_b = self.data[self.data[column_name] == group_b_value]

        if self.group_a.empty or self.group_b.empty:
            raise ValueError(
                f"One or both groups are empty after segmentation. "
                f"Group A size: {len(self.group_a)}, Group B size: {len(self.group_b)}"
            )

        print(f"Successfully segmented data. Group A size: {len(self.group_a)}, Group B size: {len(self.group_b)}")

    def perform_statistical_test(self, test_type="t-test", categorical_column=None):
        if self.group_a is None or self.group_b is None:
            raise ValueError("Groups are not segmented. Please segment data before performing the test.")
        
        if test_type == "t-test":
            if self.kpi is None:
                raise ValueError("KPI is not selected. Please select a KPI before performing the test.")
            from scipy.stats import ttest_ind
            # Perform t-test on KPI
            stat, p_value = ttest_ind(self.group_a[self.kpi], self.group_b[self.kpi], nan_policy='omit', equal_var=False)
            print(f"Performed t-test. p-value: {p_value}")
            return p_value

        elif test_type == "chi-squared":
            if categorical_column is None:
                raise ValueError("For chi-squared test, a categorical column must be provided.")
            # Create contingency table
            contingency_table = pd.crosstab(self.group_a[categorical_column], self.group_b[categorical_column])
            from scipy.stats import chi2_contingency
            stat, p_value, _, _ = chi2_contingency(contingency_table)
            print(f"Performed chi-squared test. p-value: {p_value}")
            return p_value

        else:
            raise ValueError(f"Unsupported test type '{test_type}'. Use 't-test' or 'chi-squared'.")

    def evaluate_hypothesis(self, p_value, alpha=0.05):
        if p_value < alpha:
            result = "Reject the null hypothesis: Significant differences detected."
        else:
            result = "Fail to reject the null hypothesis: No significant differences detected."
        
        # Print detailed analysis
        print(result)
        return result

    def analyze_and_report(self, hypotheses):
        """
        Analyze the outcomes and report results for each hypothesis.

        :param hypotheses: List of hypotheses to evaluate.
        """
        for hypothesis in hypotheses:
            print(f"Analyzing hypothesis: {hypothesis}")

