import argparse
from ruamel.yaml import YAML
from pathlib import Path

import polars as pl
import pandas as pd

from pysr import PySRRegressor

def calculate(path) -> None:

    config = YAML(typ="safe").load(path)

    data_frame = pl.read_csv(config["file"])

    window = config["window"]
    step = config["step"]
    offset = config["offset"]
    stop = config["stop"]

    learner = PySRRegressor(**config.get("kwargs", {}))
    learner.warm_start = False
    results = []
    while (offset + window < stop):
        print(window)
        learner.equation_file = "./test_datasetsize/" + "win_" + str(offset) + "_" + str(window) + ".csv"
        window = min(window + step, stop-offset)
        current_frame = data_frame.slice(offset,window)

        X_train = current_frame[config["features"]]
        y_train = current_frame[config["target_var"]]
        learner.fit(X_train, y_train)
        results.append([window,learner.sympy(),learner.get_best()["loss"],learner.get_best()["score"]])

    df = pd.DataFrame.from_dict(results)
    df.to_csv("datasize_logistic.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    arguments = parser.parse_args()
    calculate(arguments.config)