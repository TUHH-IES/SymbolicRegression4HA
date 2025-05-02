from learner.learner import Learner, Model

from sklearn.linear_model import LinearRegression

class LinearModel(Model):
    """
    Linear Model class for linear regression tasks.
    Inherits from the Model class.
    """

    def __init__(self, model):
        """
        Initialize the LinearModel with given parameters.

        Args:
            model (LinearRegression): The linear regression model.
        """
        self.model = model

    def predict(self, data):
        """
        Predict the output for the given input data.

        Args:
            data (DataFrame): The input data.

        Returns:
            DataFrame: The predicted output.
        """
        return self.model.predict(data)


class LinearRegressor(Learner):
    """
    Linear Regressor class for linear regression tasks.
    Inherits from the Learner class.
    """

    def __init__(
        self,
    ):
        """
        Initialize the LinearRegressor with given parameters.
        """
        self.learner = LinearRegression()

    def learnFlowFunction(self, data, inputs, target):
        """
        Learn the flow function from the given inputs and target variable.

        Args:
            data (DataFrame): The input data.
            inputs (list[str]): The input features.
            target (str): The target variable.

        Returns:
            LinearRegression: The fitted linear regressor.
        """
        self.learner.fit(data[inputs], data[target])
        return LinearModel(self.learner)
    
    def refineFlowFunction(self, data, inputs, target):
        """
        Refine the flow function from the given inputs and target variable.

        Args:
            data (DataFrame): The input data.
            inputs (list[str]): The input features.
            target (str): The target variable.

        Returns:
            LinearRegression: The fitted linear regressor.
        """
        self.learner.fit(data[inputs], data[target])
        return LinearModel(self.learner)