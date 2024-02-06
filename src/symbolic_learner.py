from typing import Any
import pandas as pd

from numpy.typing import NDArray
from gplearn.genetic import SymbolicRegressor
from pysr import PySRRegressor
#from dso import DeepSymbolicRegressor
import feyn
from sympy import sympify, simplify

import functionals
import fitness


class SymbolicLearner():
    """Wrapper class for different symbolic regressors
    """

    def __init__(self,config):
        self.type = config["type"]

        if self.type == "PySR":
            self.learner = PySRRegressor(**config.get("kwargs", {}))
            self.learner.feature_names = config["features"]
        elif self.type == "gplearn":
            self.learner = SymbolicRegressor(**config.get("kwargs", {}))
            self.learner.feature_names = config["features"]
            function_set = tuple(config["function_set"])
            function_set = function_set + tuple([getattr(functionals, name) for name in config["additional_functions"]])
            self.learner.function_set = function_set
            if "custom_metric" in config:
                self.learner.metric = getattr(fitness,config["custom_metric"])
        elif self.type == "dso":
            self.learner = DeepSymbolicRegressor(config["json_path"])
        elif self.type == "qlattice":
            self.learner = feyn.QLattice(random_seed=42)
            self.output_name = config["target_var"]
            self.feature_names = config["features"]
        else:
            raise NameError("Invalid Learner type")

    def train(self, X_train: NDArray[Any], y_train: NDArray[Any]) -> None:
        if self.type == "PySR":
            self.learner.fit(X_train,y_train)
        elif self.type == "gplearn":
            self.learner.fit(X_train,y_train)
        elif self.type == "dso":
            self.learner.fit(X_train.to_numpy(),y_train.to_numpy())
        elif self.type == "qlattice":
            X_train = pd.DataFrame(X_train, columns = self.feature_names)
            X_train[self.output_name] = y_train
            self.X_train = X_train
            self.models = self.learner.auto_run(data=X_train,output_name=self.output_name)
        else:
            raise NameError("Invalid Learner type")

    def predict(self, X_pred: NDArray[Any]) -> NDArray[Any]:
        if self.type == "PySR":
            return self.learner.predict(X_pred)
        elif self.type == "gplearn":
            return self.learner.predict(X_pred)
        elif self.type == "dso":
            return self.learner.predict(X_pred.to_numpy())
        elif self.type == "qlattice":
            best = self.models[0]
            X_pred = pd.DataFrame(X_pred, columns = self.feature_names)
            return best.predict(X_pred)
        else:
            raise NameError("Invalid Learner type")
        
    def print(self) -> None:
        if self.type == "PySR":
            print(self.learner.predict)
        elif self.type == "gplearn":
            label = f"{sympify(str(self.learner._program), locals=functionals.converter)}"
            label = simplify(label)
            print(label)
        elif self.type == "dso":
            print(self.learner.program_.pretty())
        elif self.type == "qlattice":
            self.models[0].plot(self.X_train, filename="qlattice-summary.html")
        else:
            raise NameError("Invalid Learner type")

