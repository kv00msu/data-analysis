import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class PerceptronRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate: float = 0.1, epochs: int = 10):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    @staticmethod
    def activation(z):
        return np.where(z > 0, z, 0)  # ReLU

    def fit(self, X, y):
        X_arr = X
        y_arr = y

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy()

        if isinstance(y, pd.Series):
            y_arr = y.to_numpy()

        n_features = X_arr.shape[1]

        # Initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Iterating until the number of epochs
        for epoch in range(self.epochs):

            # Traversing through the entire training set
            for i in range(X_arr.shape[0]):
                z = np.dot(X_arr, self.weights) + self.bias  # Finding the dot product and adding the bias
                y_pred = self.activation(z)  # Passing through an activation function

                # Updating weights and bias
                upd = self.learning_rate * (y_arr[i] - y_pred[i])
                self.weights += upd * X_arr[i]
                self.bias += upd

        return self

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    def score(self, X, y, sample_wight=None):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
