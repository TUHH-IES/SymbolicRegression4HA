from functools import partial
import polars as pl
import core.processed_data as processed_data

import criteria.segmentation_criteria as segmentation_criteria
from learner.learner import Learner

class Segmentor:
    """
    A class that segments a data frame into segments with differing dynamics using symbolic regression.

    Args:
        config (dict): A dictionary containing the configuration parameters for segmentation.

    Attributes:
        start_width (int): The starting width of the segmentation window.
        step_width (int): The step width for enlarging the segmentation window.
        step_iterations (int): The number of iterations for each symbolic regression run on the enlarged window.
        init_iterations (int): The number of iterations for symbolic regression on the initial window.
        hist_length (int): The history of the fitness over the enlarged windows.
        criterion (function): The fitness criterion function used for segmentation.
        selection (str): The name of the selection metric.
        learner (PySRRegressor): The symbolic regression learner.
        file_prefix (str): The prefix for the log files.
        target_var (str): The name of the target variable.

    """

    def __init__(self, config):
        self.start_width = config["start_width"]
        self.step_width = config["step_width"]
        self.criterion = getattr(segmentation_criteria, config["segmentation"]["criterion"]["name"])
        if "kwargs" in config["segmentation"]["criterion"]:
            self.criterion = partial(self.criterion, **config["segmentation"]["criterion"]["kwargs"])
        self.target = config["target_var"]
        self.inputs = config["features"]

    def segment(self, data_frame: pl.DataFrame, learner: Learner):
        """
        Perform segmentation on the given data frame.

        Args:
            data_frame (pandas.DataFrame): The data frame to be segmented.

        Returns:
            segmented_results (segmented_data.SegmentedData): The segmented data

        """
        window_size = self.start_width - self.step_width
        fit = 0.0
        fit_prev = 0.0
        while self.criterion(fit, fit_prev) and window_size < len(data_frame):
            window_size += self.step_width
            segment = data_frame.slice(0, window_size)
            fit_prev = fit
            function, fit = learner.learnFlowFunction(segment, self.inputs, self.target)
        
        return function

def buildRemainingTraces(
        traces: list[pl.DataFrame],
        accurate_segments: list[tuple[int,int]],
) -> list[pl.DataFrame]:
    """
    Build the remaining traces from the accurate segments.

    Args:
        traces (list[DataFrame]): The list of traces.
        accurate_segments (list[DataFrame]): The list of accurate segments.

    Returns:
        list[DataFrame]: The list of remaining traces.
    """
    remaining_traces = []
    for trace, segments in zip(traces, accurate_segments):
        remaining_trace = trace
        removed_size = 0
        for start, end in segments:
            start = start - removed_size
            end = end - removed_size
            removed_size += end + 1
            # Add the part before start as a new trace
            new_segment = remaining_trace.slice(0, start)
            remaining_traces.append(new_segment)
            
            # Update the remaining trace to the part after end for further iteration
            remaining_trace = remaining_trace.slice(end + 1)
        remaining_traces.append(remaining_trace)
    # Remove empty traces
    remaining_traces = [trace for trace in remaining_traces if len(trace) > 0]
        
    return remaining_traces