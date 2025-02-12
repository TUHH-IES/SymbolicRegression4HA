import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import sympy
import matplotlib.pyplot as plt

import core.processed_data as processed_data

class HybridDecisionModel:
    def __init__(self, tree, groups):
        self.tree = tree
        self.groups = groups

class ModelExtractor:
    def __init__(self, config):
        self.features = config["features"]
        self.target_var = config["target_var"]

    def createDecisionTreeModel(self, grouped_results: processed_data.GroupedData):
        """
        Create a model from the grouped results.

        Args:
            grouped_results (GroupedData): The grouped data.

        Returns:
            model: The model created from the grouped data.
        """

        data = pl.DataFrame()
        for group_id, group in grouped_results._groups.items():
            for window in group.windows:
                group_df = grouped_results.data.slice(window[0], window[1]-window[0])
                group_df = group_df.with_columns(
                    pl.Series([group_id] * len(group_df)).alias("group_id"),
                    pl.Series([group_id] * len(group_df)).alias("next_id"))
                if window[1] < len(grouped_results.data):
                    window_data = grouped_results.data.slice(window[1],1).with_columns(
                            pl.Series([group_id] * 1).alias("group_id"),
                            pl.Series([grouped_results.transitions[window[1]]] * 1).alias("next_id"))
                    group_df = group_df.vstack(window_data)
                data = data.vstack(group_df)

        clf = DecisionTreeClassifier()
        X = data[self.features] #TODO: use also previous mode as feature
        y = data["next_id"]
        clf.fit(X, y)
        print(clf.score(X, y))
        return HybridDecisionModel(clf, grouped_results)

    def evaluateDecisionTreeModel(self, model, testData):
        """
        Evaluate the model.

        Args:
            model: The model to evaluate.
            testData: The data to evaluate the model on.
        """
        #Create flow functions for all groups
        flows = {group_id: sympy.lambdify(self.features, group.equation, "numpy") for group_id, group in model.groups._groups.items()}

        # Predict next group
        predictedModes = model.tree.predict(testData[self.features]) #TODO: correct handling of initial mode -> actually the next mode is predcited

        # Use flow functions and predictedModes to predcit target value
        prediction = pl.DataFrame().with_columns(
            pl.Series(
                [flows[group_id](*testData[self.features][i])[0]
                 for i, group_id in enumerate(predictedModes)]
            ).alias(self.target_var))
        
        print(prediction[self.target_var].head())
        print(testData[self.target_var].head())
        error = metrics.mean_squared_error(testData[self.target_var], prediction[self.target_var])

        fig, ax = plt.subplots(1, 1)
        ax.plot(testData[self.target_var])
        ax.plot(prediction[self.target_var])
        plt.show()

        return error



        