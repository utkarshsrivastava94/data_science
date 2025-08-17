# src/models/baseline_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class BaselineModel:
    """
    A simple baseline model using Random Forest Regressor for time-series forecasting.
    """

    def __init__(self, df, target='nat_demand'):
        """
        Initialize the model with data and target column.

        Args:
            df (pd.DataFrame): Feature-engineered DataFrame.
            target (str): Name of the target column.
        """
        self.df = df
        self.target = target
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_test_split(self, test_size=0.2):
        """
        Splits data into train and test based on time (no shuffling).

        Args:
            test_size (float): Fraction of data to reserve for testing.

        Returns:
            X_train, X_test, y_train, y_test
        """
        split_index = int(len(self.df) * (1 - test_size))
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def train(self, X_train, y_train):
        """
        Fit the model on training data.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data and print metrics.
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print("Evaluation Metrics:")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2_score(y_test, y_pred):.3f}")
        
        return y_test, y_pred


    def predict(self, X):
        """
        Predict using the trained model.
        """
        return self.model.predict(X)
