from functools import partial
from pysr import PySRRegressor
from collections import deque
import polars as pl
import core.processed_data as processed_data

import segmentation_criteria

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
        self.step_iterations = config["step_iterations"]
        self.init_iterations = config["segmentation"]["kwargs"]["niterations"]
        self.hist_length = config["hist_length"]
        self.criterion = getattr(segmentation_criteria, config["segmentation"]["criterion"]["name"])
        if "kwargs" in config["segmentation"]["criterion"]:
            self.criterion = partial(self.criterion, **config["segmentation"]["criterion"]["kwargs"])
        if "selection" not in config:
            config["selection"] = "loss"
        self.selection = config["selection"]
        self.learner = PySRRegressor(**config["segmentation"].get("kwargs", {}))
        self.learner.feature_names = config["features"]
        self.file_prefix = config["file_prefix"]
        self.target_var = config["target_var"]

    def segment(self, data_frame):
        """
        Perform segmentation on the given data frame.

        Args:
            data_frame (pandas.DataFrame): The data frame to be segmented.

        Returns:
            segmented_results (segmented_data.SegmentedData): The segmented data

        """
        fitness_hist = deque([], self.hist_length)
        switches = [0]
        window = [0, self.start_width - self.step_width]
        segments = pl.DataFrame({
            "window_start": pl.Series(dtype=pl.Int64, values=[]),
            "window_end": pl.Series(dtype=pl.Int64, values=[]),
            "extensions": pl.Series(dtype=pl.Int64, values=[]),
            "equation": pl.Series(dtype=pl.Utf8, values=[]),
            self.selection: pl.Series(dtype=pl.Float64, values=[])
        })

        while window[1] < len(data_frame):
            self.learner.warm_start = False
            fitness_hist = deque([], self.hist_length)
            extension = 0
            while len(fitness_hist) < 2 or (
                self.criterion(fitness_hist) and window[1] < len(data_frame)
            ):
                self.learner.equation_file = (
                    "./equations/"
                    + self.file_prefix
                    + "_win"
                    + str(len(switches))
                    + "_ext"
                    + str(extension)
                    + ".csv"
                )
                if hasattr(self.learner, "equations_"):
                    best_equation = self.learner.sympy()
                window[1] += self.step_width
                window[1] = min(window[1], len(data_frame))

                print(window)
                current_frame = data_frame.slice(window[0], (window[1] - window[0]))

                X_train = current_frame[self.learner.feature_names]
                y_train = current_frame[self.target_var]
                self.learner.fit(X_train, y_train)
                fitness_hist.append(self.learner.get_best()[self.selection])
                self.learner.warm_start = True
                self.learner.niterations = self.step_iterations
                extension = extension + 1

            if(window[1] >= len(data_frame)):
                result_row = pl.DataFrame({
                    "window_start": [window[0]],
                    "window_end": [len(data_frame)],
                    "extensions": [extension],
                    "equation": [str(best_equation)],
                    self.selection: [self.learner.get_best()[self.selection]]
                })
                segments = segments.vstack(result_row)
                break

            window_end = window[1] - self.step_width
            result_row = pl.DataFrame({
                "window_start": [window[0]],
                "window_end": [window_end],
                "extensions": [extension - 1],
                "equation": [str(best_equation)],
                self.selection: [self.learner.get_best()[self.selection]]
            })
            segments = segments.vstack(result_row)
            switches.append(window_end)

            window[0] = window[1] - self.step_width
            window[1] = min(window[0] + self.start_width - self.step_width, len(data_frame))
            self.learner.niterations = self.init_iterations

        segmented_results = processed_data.SegmentedData(data_frame, segments, switches, self.target_var)
        
        return segmented_results
