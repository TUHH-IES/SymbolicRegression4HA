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
    
    def to_string(self):
        """
        Convert the model to a string representation.

        Returns:
            str: The string representation of the model.
        """
        return f"Linear Model - Coefficients: {self.model.coef_}, Intercept: {self.model.intercept_}"


class LinearRegressor(Learner):
    """
    Linear Regressor class for linear regression tasks.
    Inherits from the Learner class.
    """

    def __init__(
        self,
        kwargs=None,
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
        predictions = self.learner.predict(data[inputs])
        #mse = mean_squared_error(data[target], predictions)
        max_error = (abs(data[target] - predictions)).max()
        return LinearModel(self.learner), max_error
    
    def refineFlowFunction(self, data, inputs, target) -> LinearModel:
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