from pathlib import Path
from typing import Any, cast

from numpy.typing import NDArray
from gplearn.genetic import SymbolicRegressor


class SymbolicLearner():
    """Wrapper class for gplearn's SymbolicRegressor.

    Reference: https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-regressor
    """

    regressor: SymbolicRegressor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.regressor = SymbolicRegressor(*args, **kwargs)

    def train(self, inputs: NDArray[Any], outputs: NDArray[Any]) -> None:
        self.regressor.fit(inputs, outputs)

    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], self.regressor.predict(inputs))

    def save(self, path: Path) -> None:
        graph_str = self.regressor._program.export_graphviz()

    def load(self, path: Path) -> None:
        return
