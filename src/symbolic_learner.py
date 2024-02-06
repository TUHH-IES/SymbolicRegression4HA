from typing import Any

from numpy.typing import NDArray
from pysr import PySRRegressor


class SymbolicLearner():
    """Wrapper class for PySR symbolic regressor
    """

    def __init__(self,config):
        self.learner = PySRRegressor(**config.get("kwargs", {}))
        self.learner.feature_names = config["features"]

    def train(self, X_train: NDArray[Any], y_train: NDArray[Any]) -> None:
        self.learner.fit(X_train,y_train)

    def predict(self, X_pred: NDArray[Any]) -> NDArray[Any]:
        return self.learner.predict(X_pred)
        
    def print(self) -> None:
        print(self.learner.predict)

