
from abc import ABC, abstractmethod

import numpy as np
from polars import DataFrame, concat


class Learner(ABC):
    
    @abstractmethod
    def learnFlowFunction(self, data: DataFrame, inputs: list[str], target: str) -> None:
        """
        Learn the flow function from the given inputs and target variable.

        Args:
            inputs (DataFrame): The input data.
            target (str): The target variable.
        """
        pass

    @abstractmethod
    def refineFlowFunction(self, data: DataFrame, inputs: list[str], target: str) -> None:
        """
        Refine the flow function from the given inputs and target variable.

        Args:
            inputs (DataFrame): The input data.
            target (str): The target variable.
        """
        pass

class Model(ABC):
    """
    Abstract base class for models.
    """
    
    @abstractmethod
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Predict the output for the given input data.

        Args:
            data (DataFrame): The input data.

        Returns:
            DataFrame: The predicted output.
        """
        pass

    @abstractmethod
    def to_string(self) -> str:
        """
        Convert the model to a string representation.

        Returns:
            str: The string representation of the model.
        """
        pass


def getAccurateSegments(
        traces: list[DataFrame],
        model: Model,
        inputs: list[str],
        target: str,
        threshold: float,
):
    """
    Get accurate segments from the traces using the given model.
    """
    accurate_segments = []
    all_accurate_data = []

    for trace in traces:
        predictions = model.predict(trace[inputs])
        errors = np.abs(predictions - trace[target])
        local_segments = []

        start = None
        for i, error in enumerate(errors):
            if np.all(error < threshold):
                if start is None:
                    start = i  # Start of a new interval
            else:
                if start is not None:
                    local_segments.append((start, i - 1))  # End of the interval
                    all_accurate_data.append(trace[start:i])  # Collect data for the segment
                    start = None
        if start is not None:
            local_segments.append((start, len(errors) - 1))  # Handle last interval
            all_accurate_data.append(trace[start:len(errors)])  # Collect data for the last segment
        accurate_segments.append(local_segments)

    # Combine all accurate data into a single DataFrame
    combined_accurate_data = concat(all_accurate_data, how="vertical")


    return accurate_segments, combined_accurate_data
        