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

    def plot_premium_claims_correlation(self, premium_col, claims_col, category_col):
            """
            Plot the relationship between total premiums and total claims by category.
            :param premium_col: Column name for total premiums.
            :param claims_col: Column name for total claims.
            :param category_col: Column name for categorical segmentation.
            """
            if premium_col not in self.df.columns or claims_col not in self.df.columns:
                raise ValueError("Specified columns do not exist in the DataFrame.")
            if not pd.api.types.is_numeric_dtype(self.df[premium_col]) or not pd.api.types.is_numeric_dtype(self.df[claims_col]):
                raise ValueError("Both premium and claims columns must be numerical.")

            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=self.df, x=premium_col, y=claims_col, hue=category_col, style=category_col, palette='Set2', s=100)
            sns.regplot(data=self.df, x=premium_col, y=claims_col, scatter=False, color='gray', ci=None)
            plt.title(f"Premium vs. Claims Correlation by {category_col}", fontsize=16)
            plt.xlabel(premium_col, fontsize=14)
            plt.ylabel(claims_col, fontsize=14)
            plt.legend(title=category_col, fontsize=12)
            plt.tight_layout()
            plt.show()
            plt.show()
    def plot_monthly_trends(self, month_col, category_col):
            """
            Plot monthly trends in insurance categories.
            :param month_col: Column name for transaction months.
            :param category_col: Column name for categories.
            """
            if month_col not in self.df.columns or category_col not in self.df.columns:
                raise ValueError("Specified columns do not exist in the DataFrame.")

            monthly_data = self.df.groupby([month_col, category_col]).size().reset_index(name='Count')
            plt.figure(figsize=(14, 8))
            sns.lineplot(data=monthly_data, x=month_col, y='Count', hue=category_col, marker='o')
            plt.title(f"Monthly Trends in {category_col}", fontsize=16)
            plt.xlabel(month_col, fontsize=14)
            plt.ylabel("Number of Policies", fontsize=14)
            plt.legend(title=category_col, fontsize=12)
            plt.tight_layout()
            plt.show()