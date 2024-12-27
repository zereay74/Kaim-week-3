import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, dataframe):
        """
        Initialize the DataAnalyzer with a DataFrame.
        :param dataframe: pandas DataFrame to be analyzed.
        """
        self.df = dataframe

    def identify_column_types(self):
        """
        Identify numerical and categorical columns in the DataFrame.
        :return: A dictionary with keys 'numerical' and 'categorical'.
        """
        numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        result = {'numerical': numerical_cols, 'categorical': categorical_cols}

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in result.items()]))
    
        return df
        #return {'numerical': numerical_cols,  'categorical': categorical_cols}

    def plot_univariate_distribution(self, columns):
        """
        Plot histograms for specified numerical columns.
        :param columns: List of numerical column names to plot.
        """
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise ValueError(f"Column '{column}' is not numerical.")

            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[column], kde=True, bins=30)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

    def plot_bivariate_relationship(self, x, y, hue=None):
        """
        Explore relationships between two columns using a scatter plot.
        :param x: Name of the x-axis column.
        :param y: Name of the y-axis column.
        :param hue: (Optional) Name of the column to color-code the data points.
        """
        if x not in self.df.columns or y not in self.df.columns:
            raise ValueError("Specified columns do not exist in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(self.df[x]) or not pd.api.types.is_numeric_dtype(self.df[y]):
            raise ValueError("Both x and y columns must be numerical.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue)
        plt.title(f"Relationship between {x} and {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_correlation_matrix(self, columns=None):
        """
        Plot a correlation matrix for specified numerical columns.
        :param columns: List of numerical columns to include in the correlation matrix. If None, use all numerical columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns.tolist()

        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        correlation_matrix = self.df[columns].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
    def plot_trends_over_geography(self, numerical_column, categorical_columns):
        """
        Compare trends of a numerical column across multiple categorical columns.
        :param numerical_column: Name of the numerical column (e.g., CalculatedPremiumPerTerm).
        :param categorical_columns: List of categorical columns (e.g., Make, CoverType).
        """
        if numerical_column not in self.df.columns:
            raise ValueError(f"Numerical column '{numerical_column}' does not exist in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(self.df[numerical_column]):
            raise ValueError(f"Column '{numerical_column}' is not numerical.")

        for cat_col in categorical_columns:
            if cat_col not in self.df.columns:
                raise ValueError(f"Categorical column '{cat_col}' does not exist in the DataFrame.")
            if not pd.api.types.is_categorical_dtype(self.df[cat_col]) and not pd.api.types.is_object_dtype(self.df[cat_col]):
                raise ValueError(f"Column '{cat_col}' is not categorical.")

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.df, x=cat_col, y=numerical_column)
            plt.title(f"Trends of {numerical_column} by {cat_col}")
            plt.xlabel(cat_col)
            plt.ylabel(numerical_column)
            plt.xticks(rotation=45)
            plt.show()
'''
# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'ZipCode': [1001, 1002, 1003, 1004, 1005],
        'TotalPremium': [200, 150, 300, 250, 400],
        'TotalClaims': [180, 120, 280, 230, 380],
        'Region': ['East', 'East', 'West', 'West', 'South']
    }
    df = pd.DataFrame(data)

    # Initialize the analyzer
    analyzer = DataAnalyzer(df)

    # Identify column types
    column_types = analyzer.identify_column_types()
    print("Column Types:", column_types)

    # Univariate analysis
    print("\nUnivariate Analysis:")
    analyzer.plot_univariate_distribution(['TotalPremium', 'TotalClaims'])

    # Bivariate relationship
    print("\nBivariate Analysis:")
    analyzer.plot_bivariate_relationship('TotalPremium', 'TotalClaims', hue='ZipCode')

    # Correlation matrix
    print("\nCorrelation Matrix:")
    analyzer.plot_correlation_matrix(['TotalPremium', 'TotalClaims'])
'''