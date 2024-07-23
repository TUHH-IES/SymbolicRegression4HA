from functools import partial
from pysr import PySRRegressor
import polars as pl
from statistics import mean

from core.processed_data import GroupedData

import grouping_criteria


class GroupIdentificator:
    def __init__(self, config):
        """
        Initializes a GroupIdentificator object.

        Args:
            config (dict): Configuration parameters for the GroupIdentificator.

        Attributes:
            criterion (function): The grouping criterion function.
            selection (str): The selection criteria for grouping.
            learner (PySRRegressor): The PySRRegressor object for learning equations.
            file_prefix (str): The file prefix for equation files.
        """
        self.criterion = getattr(
            grouping_criteria, config["grouping"]["criterion"]["name"]
        )
        if "kwargs" in config["grouping"]["criterion"]:
            self.criterion = partial(
                self.criterion, **config["grouping"]["criterion"]["kwargs"]
            )
        if "selection" not in config:
            config["selection"] = "loss"
        self.selection = config["selection"]
        self.learner = PySRRegressor(**config["grouping"].get("kwargs", {}))
        self.learner.warm_start = False
        self.learner.feature_names = config["features"]
        self.file_prefix = config["file_prefix"]

    def _set_learner_log_file(self, window, group):
        """
        Sets the equation file path for the learner.

        Args:
            window (list): The window range.
            group (int): The group ID.
        """
        self.learner.equation_file = (
            "./equations/"
            + self.file_prefix
            + "_win"
            + str(window[1])
            + "_group"
            + str(group)
            + ".csv"
        )

    def group_segments(self, segmented_results):
        """
        Groups the segments using symbolic regression and a grouping criterion.

        Args:
            segmented_results (SegmentedResults): The segmented results.

        Returns:
            GroupedData: The grouped data.
        """
        # todo: use previous models as starting point (option:
        # 1) try previous models for both. If one of them is better than the two before, a new one is found,
        # 2) test fit first, then re-learn?)
        # alternative procedure: test against all groups and choose the smallest one, if the error is below something or the increase in accuracy is large enough

        segments = segmented_results.segments
        data_frame = segmented_results.data
        target_var = segmented_results.target_var

        group_data = GroupedData(data_frame, target_var)
        for segment in segments.iter_rows(named=True):
            window = [segment["window_start"], segment["window_end"]]
            curr_segment_loss = segment[self.selection]
            print("Current window:", window)

            df_window = data_frame.slice(window[0], (window[1] - window[0]))
            if not group_data.groups:
                X_train = df_window[self.learner.feature_names]
                y_train = df_window[target_var]
                self.learner.fit(X_train, y_train)

                group_data.create_group(
                    df_window,
                    self.learner.sympy(),
                    window,
                    curr_segment_loss,
                    [curr_segment_loss],
                )
                continue
            else:
                found_group = False
                for group in group_data.groups:
                    print(
                        "Current group", group.group_id, "of", len(group_data.groups)
                    )
                    self._set_learner_log_file(window, group.group_id)

                    concatenation = pl.concat([group.data, df_window])
                    X_train = concatenation[self.learner.feature_names]
                    y_train = concatenation[target_var]
                    self.learner.fit(X_train, y_train)
                    equation = self.learner.sympy()
                    loss = self.learner.get_best()[self.selection]
                    if self.criterion(
                        mean(group.segment_losses),  # todo: weighted by segment length?
                        loss,
                    ):
                        print("group", window, "into", group.group_id)
                        group.append_segment(
                            df_window, equation, window, loss, curr_segment_loss
                        )
                        found_group = True
                        break

                if not found_group:
                    print("Create new group")
                    X_train = df_window[self.learner.feature_names]
                    y_train = df_window[target_var]
                    self.learner.fit(X_train, y_train)

                    group_data.create_group(
                        df_window,
                        self.learner.sympy(),
                        window,
                        curr_segment_loss,
                        [curr_segment_loss],
                    )

        group_data.print_groups()
        return group_data
