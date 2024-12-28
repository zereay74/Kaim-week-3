import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

class InsuranceDataProcessor:
    def __init__(self, dataframe):
        self.data = dataframe

    def feature_engineering(self):
        """Create new features related to TotalPremium and TotalClaims."""
        self.data['PremiumToClaimsRatio'] = self.data['TotalPremium'] / (self.data['TotalClaims'] + 1)
        self.data['ClaimsToPremiumRatio'] = self.data['TotalClaims'] / (self.data['TotalPremium'] + 1)
        self.data['SumInsuredToPremium'] = self.data['SumInsured'] / (self.data['TotalPremium'] + 1)
        return self.data

    def encode_categorical_data(self):
        """Convert categorical columns to numeric using encoding."""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        return self.data

    def train_test_split(self, target, test_size=0.3, random_state=42):
        """Split the dataset into train and test sets."""
        X = self.data.drop(columns=[target])
        y = self.data[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def build_model(self, model_type, X_train, y_train):
        """Train a model based on the specified type."""
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        elif model_type == 'xgboost':
            model = XGBRegressor(random_state=42)
        else:
            raise ValueError("Unsupported model type")
        
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model using various metrics."""
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'R^2 Score': r2
        }

    def feature_importance(self, model, feature_names):
        """Analyze feature importance (for tree-based models)."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        else:
            raise ValueError("Feature importance is not supported for this model")

    def explain_predictions(self, model, X_train):
        """Use SHAP for model interpretability."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        return explainer, shap_values
''' 
# Example Usage
if __name__ == "__main__":
    # Load example dataset
    data = pd.DataFrame({  # Replace with actual data loading
        'TotalPremium': np.random.rand(100) * 1000,
        'TotalClaims': np.random.rand(100) * 500,
        'SumInsured': np.random.rand(100) * 10000,
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })

    processor = InsuranceDataProcessor(data)

    # Feature Engineering
    data = processor.feature_engineering()

    # Encoding Categorical Data
    data = processor.encode_categorical_data()

    # Train-Test Split
    X_train, X_test, y_train, y_test = processor.train_test_split(target='TotalPremium')

    # Model Building and Evaluation
    models = ['linear_regression', 'random_forest', 'xgboost']
    for model_type in models:
        model = processor.build_model(model_type, X_train, y_train)
        metrics = processor.evaluate_model(model, X_test, y_test)
        print(f"{model_type} metrics: {metrics}")

        # Feature Importance (for tree-based models)
        if model_type in ['random_forest', 'xgboost']:
            importance = processor.feature_importance(model, X_train.columns)
            print(f"{model_type} feature importance: {importance}")

        # SHAP Explanation
        if model_type in ['xgboost']:
            explainer, shap_values = processor.explain_predictions(model, X_train)
            shap.summary_plot(shap_values, X_train)
'''