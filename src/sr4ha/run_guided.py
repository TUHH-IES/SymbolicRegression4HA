import argparse
from matplotlib import pyplot as plt
from ruamel.yaml import YAML
from pathlib import Path
import polars as pl
import numpy as np
import time

import core.segmentor

from learner.learner import Model, getAccurateSegments
from learner.linear_regressor import LinearRegressor

def predictFromModes(modes: list[Model], data_frame : pl.DataFrame, features, target_var, mode_var):
    """
    Predict the target variable using the learned modes.

    Args:
        modes (list): List of learned modes.
        data_frame (DataFrame): The input data frame.
        features (list): List of feature names.
        target_var (str): The target variable name.
        mode_var (str): The mode variable name.

    Returns:
        DataFrame: The predicted data frame with the target variable.
    """
    predictions = []
    for row in data_frame.iter_rows(named=True):
        mode_index = int(row[mode_var])
        if mode_index < len(modes):
            mode = modes[mode_index]
            row_df = pl.DataFrame([row])
            prediction = mode.predict(row_df[features])
            predictions.append(prediction[0])
        else:
            predictions.append(None)  # Handle cases where mode index is invalid

    return pl.DataFrame({target_var: predictions, 't': data_frame['t']})

def main(path):
    modes = []

    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(
        config["file"], schema_overrides=[pl.Float64] * len(config["features"])
    )

    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        config["target_var"] = "diff"

    traces = [data_frame]
    trajectory = traces[0]

    start_time = time.time()
    while len(trajectory) > 0:
        # Segmentation
        segmentor = core.segmentor.Segmentor(config)
        learner = LinearRegressor()
        model = segmentor.segment(trajectory, learner)

        accurate_segments, mode_data = getAccurateSegments(traces, model, config["features"], config["target_var"], config["threshold"])

        model = learner.refineFlowFunction(mode_data, config["features"], config["target_var"])
        modes.append(model)

        # Build remaining traces
        traces = core.segmentor.buildRemainingTraces(traces, accurate_segments)
        if len(traces) == 0:
            break

        trajectory = traces[0]

    end_time = time.time()
    for i, mode in enumerate(modes):
        print(f"Mode {i}:")
        print(mode.to_string())

    predictions = predictFromModes(modes, data_frame, config["features"], config["target_var"], config["mode_var"])
    target = data_frame[config["target_var"]]

    predictions_array = np.array(predictions[config["target_var"]])
    target_array = np.array(target)
    mse = np.mean((predictions_array - target_array) ** 2)
    print(f"Mean Squared Error: {mse}")

    learn_time = end_time - start_time
    with open("metrics.txt", "w") as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Learning Time (s): {learn_time}\n")
    
    predictions['t', config["target_var"]].write_csv('simple-linear-pred.csv', include_header=False)
    data_frame['t', config["target_var"]].write_csv('simple-linear-gt.csv', include_header=False)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(predictions[config["target_var"]], label="Predictions")
    plt.plot(target, label="Target")
    plt.xlabel('y')
    plt.ylabel('t')
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    arguments = parser.parse_args()
    main(arguments.config)
