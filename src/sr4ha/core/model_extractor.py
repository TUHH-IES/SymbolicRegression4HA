import polars as pl
from sklearn import tree
from sklearn import metrics
import sympy
import matplotlib.pyplot as plt

import core.processed_data as processed_data

class HybridDecisionModel:
    def __init__(self, tree: tree.DecisionTreeClassifier, groupedData: processed_data.GroupedData):
        self.tree: tree.DecisionTreeClassifier = tree
        self.groupedData: processed_data.GroupedData = groupedData

class ModelExtractor:
    def __init__(self, config):
        self.features = config["features"]
        self.target_var = config["target_var"]

    def createDecisionTreeModel(self, grouped_results: processed_data.GroupedData, with_prev_id: bool = False):
        """
        Create a model from the grouped results.

        Args:
            grouped_results (GroupedData): The grouped data.

        Returns:
            model: The model created from the grouped data.
        """

        # Add prev_id as feature
        data = pl.DataFrame()
        for group_id, group in grouped_results._groups.items():
            for window in group.windows:
                group_df = grouped_results.data.slice(window[0]+1, window[1]-window[0]+1)
                group_df = group_df.with_columns(
                    pl.Series([group_id] * len(group_df)).alias("group_id"),
                    pl.Series([group_id] * len(group_df)).alias("prev_id"))
                if window[0] > 0:
                    window_data = grouped_results.data.slice(window[0],1).with_columns(
                            pl.Series([group_id] * 1).alias("group_id"),
                            pl.Series([grouped_results.transitions[window[0]]] * 1).alias("prev_id"))
                    group_df = group_df.vstack(window_data)
                data = data.vstack(group_df)

        clf = tree.DecisionTreeClassifier()
        if with_prev_id:
            dt_features = self.features + ["prev_id"]
        else:
            dt_features = self.features
        X = data[dt_features] #TODO: use also previous mode as feature
        y = data["group_id"]
        clf.fit(X, y)
        tree.plot_tree(clf)
        return HybridDecisionModel(clf, grouped_results)

    def evaluateDecisionTreeModel(self, model, testData, visualize: bool = True):
        """
        Evaluate the model.

        Args:
            model: The model to evaluate.
            testData: The data to evaluate the model on.
        """
        #Create flow functions for all groups
        flows = {group_id: sympy.lambdify(self.features, group.equation, "numpy") for group_id, group in model.groupedData._groups.items()}

        # Predict next group
        predictedModes = model.tree.predict(testData[self.features])

        # Use flow functions and predictedModes to predict target value
        prediction = pl.DataFrame().with_columns(
            pl.Series(
                [flows[group_id](*testData[self.features][i])[0]
                 for i, group_id in enumerate(predictedModes)]
            ).alias(self.target_var))
        error = metrics.mean_squared_error(testData[self.target_var], prediction[self.target_var])

        if visualize:
            fig, ax = plt.subplots(1, 1)
            ax.plot(testData[self.target_var])
            ax.plot(prediction[self.target_var])
            plt.show()

        return error

    def evaluateWithPrevMode(self, model, testData, initialMode, visualize: bool = True):
        """
        Evaluate the model.

        Args:
            model: The model to evaluate.
            testData: The data to evaluate the model on.
        """
        nextMode = initialMode
        predictedModes = []
        for i in range(len(testData)):
            # Predict next group
            row = testData[self.features].slice(i, 1).with_columns(pl.Series([nextMode]).alias("prev_id"))
            dt_features = self.features + ["prev_id"]
            predictedMode = model.tree.predict(row[dt_features])
            predictedModes.append(predictedMode[0])
            nextMode = predictedMode

        flows = {group_id: sympy.lambdify(self.features, group.equation, "numpy") for group_id, group in model.groupedData._groups.items()}
        # Use flow functions and predictedModes to predict target value
        prediction = pl.DataFrame().with_columns(
            pl.Series(
                [flows[group_id](*testData[self.features][i])[0]
                 for i, group_id in enumerate(predictedModes)]
            ).alias(self.target_var))
        error = metrics.mean_squared_error(testData[self.target_var], prediction[self.target_var])

        if visualize:
            fig, ax = plt.subplots(1, 1)
            ax.plot(testData[self.target_var])
            ax.plot(prediction[self.target_var])
            plt.show()

        return error