
from pysr import PySRRegressor
from learner.learner import Learner


class SymbolicRegressor(Learner):
    """
    Symbolic Regressor class for symbolic regression tasks.
    Inherits from the Learner class.
    """

    def __init__(self, config):
        """
        Initialize the SymbolicRegressor with given parameters.

        Args:
            **kwargs: Additional parameters for the symbolic regressor.
        """
        super().__init__()
        self.step_iterations = config["step_iterations"]
        self.init_iterations = config["segmentation"]["kwargs"]["niterations"]
        if "selection" not in config:
            config["selection"] = "loss"
        self.selection = config["selection"]
        self.learner = PySRRegressor(**config["segmentation"].get("kwargs", {}))
        self.learner.feature_names = config["features"]
        self.file_prefix = config["file_prefix"]
        self.learner.warm_start = False

    def learnFlowFunction(self, data, inputs, target):
        """
        Learn the flow function from the given inputs and target variable.

        Args:
            data (DataFrame): The input data.
            inputs (list[str]): The input features.
            target (str): The target variable.

        Returns:
            PySRRegressor: The fitted symbolic regressor.
        """
        self.learner.fit(data[inputs], data[target])
        return self.learner
    
    def refineFlowFunction(self, data, inputs, target):
        """
        Refine the flow function from the given inputs and target variable.

        Args:
            data (DataFrame): The input data.
            inputs (list[str]): The input features.
            target (str): The target variable.

        Returns:
            PySRRegressor: The fitted symbolic regressor.
        """
        self.learner.fit(data[inputs], data[target])
        return self.learner