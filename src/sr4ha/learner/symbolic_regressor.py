
from pysr import PySRRegressor
import sympy
from learner.learner import Learner, Model

class SymbolicModel(Model):
    """
    Symbolic Model class for symbolic regression tasks.
    Inherits from the Model class.
    """

    def __init__(self, model: PySRRegressor):
        """
        Initialize the SymbolicModel with given parameters.

        Args:
            model (PySRRegressor): The symbolic regression model.
        """
        self.model: PySRRegressor = model

    def predict(self, data):
        """
        Predict the output for the given input data.

        Args:
            data (DataFrame): The input data.

        Returns:
            DataFrame: The predicted output.
        """
        return self.model.predict(data)

    def to_string(self):
        """
        Convert the model to a string representation.

        Returns:
            str: The string representation of the model.
        """
        return sympy.sstr(self.model.sympy())

class SymbolicRegressor(Learner):
    """
    Symbolic Regressor class for symbolic regression tasks.
    Inherits from the Learner class.
    """

    def __init__(self, kwargs):
        """
        Initialize the SymbolicRegressor with given parameters.

        Args:
            **kwargs: Additional parameters for the symbolic regressor.
        """
        super().__init__()
        self.learner = PySRRegressor(**kwargs)
        self.learner.warm_start = True

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
        predictions = self.learner.predict(data[inputs])
        #mse = mean_squared_error(data[target], predictions)
        max_error = (abs(data[target] - predictions)).max()
        self.learner.warm_start = True
        return SymbolicModel(self.learner), max_error
    
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
        self.learner.warm_start = False
        self.learner.fit(data[inputs], data[target])
        return SymbolicModel(self.learner)