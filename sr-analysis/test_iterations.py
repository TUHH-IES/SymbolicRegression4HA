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
    offset = config["offset"]
    data_frame = data_frame.slice(offset,window)
    X_train = data_frame[config["features"]]
    y_train = data_frame[config["target_var"]]

    step = config["kwargs"]["niterations"]
    stop = config["stop"]

    learner = PySRRegressor(**config.get("kwargs", {}))
    learner.warm_start = True
    results = []
    iterations = config["kwargs"]["niterations"]
    while (iterations < stop):
        learner.equation_file = "./test_iterations/" + "it_" + str(iterations) + ".csv"

        learner.fit(X_train, y_train)
        results.append([iterations,learner.sympy(),learner.get_best()["loss"],learner.get_best()["score"]])
        print(iterations,learner.get_best()["score"])
        iterations = iterations + step

        df = pd.DataFrame.from_dict(results)
        df.to_csv("iterations_logistic.csv")

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