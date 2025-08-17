# src/data_prep/feature_engineering.py

import pandas as pd

class FeatureEngineer:
    """
    A class to perform time-series feature engineering for load forecasting.

    Attributes:
        df (pd.DataFrame): A copy of the original dataframe with datetime index.
    """

    def __init__(self, df):
        """
        Initialize the FeatureEngineer with a copy of the input DataFrame.

        Args:
            df (pd.DataFrame): Input dataframe with 'nat_demand' and datetime index.
        """
        self.df = df.copy()

    def add_time_features(self):
        """
        Adds basic time-based features:
            - Hour of day
            - Day of week
            - Month
            - Weekend flag

        Returns:
            self: Enables method chaining
        """
        self.df['hour'] = self.df.index.hour
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month
        self.df['weekend'] = self.df['dayofweek'].isin([5, 6]).astype(int)
        return self

    def add_lag_features(self, lags=[1, 24, 168]):
        """
        Adds lag features for the target variable `nat_demand`.

        Args:
            lags (list): Time lags (in hours) to include. For example:
                         - 1 (previous hour)
                         - 24 (same hour previous day)
                         - 168 (same hour previous week)

        Returns:
            self: Enables method chaining
        """
        for lag in lags:
            self.df[f'nat_demand_lag_{lag}'] = self.df['nat_demand'].shift(lag)
        return self

    def add_rolling_features(self, windows=[3, 24, 168]):
        """
        Adds rolling mean and standard deviation features for the target.

        Args:
            windows (list): Window sizes (in hours) for rolling statistics. For example:
                            - 3: 3-hour rolling window
                            - 24: 1-day rolling window
                            - 168: 1-week rolling window

        Returns:
            self: Enables method chaining
        """
        for window in windows:
            # Use shifted data to avoid data leakage
            self.df[f'nat_demand_rollmean_{window}'] = (
                self.df['nat_demand'].shift(1).rolling(window).mean()
            )
            self.df[f'nat_demand_rollstd_{window}'] = (
                self.df['nat_demand'].shift(1).rolling(window).std()
            )
        return self

    def finalize(self):
        """
        Drops rows with NaN values caused by lag and rolling features.

        Returns:
            pd.DataFrame: Final cleaned and feature-rich dataframe ready for modeling.
        """
        self.df.dropna(inplace=True)
        return self.df
