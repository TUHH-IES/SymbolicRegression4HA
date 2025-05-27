import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from learner.learner import Learner, Model

class LinearDifferentialModel(Model):
    def __init__(self, model):
        self.model: LinearRegression = model

    def predict(self, data):
        return self.model.predict(data)

    def to_string(self):
        return f"Linear Differential Model - Coefficients: {self.model.coef_}, Intercept: {self.model.intercept_}"

class LinearDifferentialLearner(Learner):
    def __init__(self, kwargs = None):
        self.learner = LinearRegression(**kwargs, fit_intercept=True)

    def learnFlowFunction(self, data, inputs, target):
        X_t, X_tp1 = create_windows(data, target)
        self.learner.fit(X_t, X_tp1)
        model = LinearDifferentialModel(self.learner)
        preds = self.learner.predict(X_t)
        max_error = np.max(np.abs(preds - X_tp1.to_numpy()))
        return model, max_error

    def refineFlowFunction(self, data, inputs, target):
        return self.learnFlowFunction(data, inputs, target)[0]
    
def create_windows(df: pl.DataFrame, features: list[str]):
    X_t = df[features].slice(0, len(df) - 1)
    X_tp1 = df[features].slice(1, len(df) - 1)
    return X_t, X_tp1