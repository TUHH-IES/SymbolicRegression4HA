import argparse
from ruamel.yaml import YAML
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
import copy
from functools import partial
from statistics import mean

from pysr import PySRRegressor

import criteria


def identify_switch(config, data_frame):

    # todo: add an 'auto'-option which estimates this from data
    start_width = config["start_width"]
    step_width = config["step_width"]
    step_iterations = config["step_iterations"]
    hist_length = config["hist_length"]
    criterion = getattr(criteria, config["criterion"]["name"])
    if "kwargs" in config["criterion"]:
        criterion = partial(criterion, **config["criterion"]["kwargs"])
    if not "selection" in config:
        config["selection"] = "loss"
    selection = config["selection"]

    learner = PySRRegressor(**config.get("kwargs", {}))
    learner.feature_names = config["features"]

    fitness_hist = deque([], hist_length)
    switches = [0]
    window = [0, start_width - step_width]
    log_list = []
    while window[1] < len(data_frame):
        learner.warm_start = False
        fitness_hist = deque([], hist_length)
        extension = 0
        while len(fitness_hist) < 2 or (
            criterion(fitness_hist) and window[1] < len(data_frame)
        ):
            learner.equation_file = (
                "./equations/"
                + config["file_prefix"]
                + "_win"
                + str(len(switches))
                + "_ext"
                + str(extension)
                + ".csv"
            )
            if hasattr(learner, "equations_"):
                best_equation = learner.sympy()
            window[1] += step_width
            window[1] = min(window[1], len(data_frame))

            print(window)
            current_frame = data_frame.slice(window[0], (window[1] - window[0]))

            X_train = current_frame[config["features"]]
            y_train = current_frame[config["target_var"]]
            learner.fit(X_train, y_train)
            fitness_hist.append(learner.get_best()[selection])
            learner.warm_start = True
            learner.niterations = step_iterations
            extension = extension + 1
            log_best = []
            log_best.append(
                [
                    window,
                    learner.sympy(),
                    learner.get_best()["loss"],
                    learner.get_best()["score"],
                ]
            )
            df_log = pd.DataFrame.from_dict(log_best)
            df_log.to_csv(
                "./equations/"
                + config["file_prefix"]
                + "_win"
                + str(len(switches))
                + "_ext"
                + str(extension)
                + "_best.csv"
            )

        log = dict()
        log["extensions"] = extension - 1
        log["window"] = copy.deepcopy(window)
        log["window"][1] = log["window"][1] - step_width
        log["equation"] = best_equation
        log[selection] = learner.get_best()[selection]
        log_list.append(log)
        switches.append(window[1] - step_width)
        window[0] = window[1] - step_width
        window[1] = min(window[0] + start_width - step_width, len(data_frame))
        learner.niterations = config["kwargs"]["niterations"]

    switches[-1] = len(data_frame)
    log_list[-1]["window"][1] = len(data_frame)
    log_list[-1]["extensions"] = log_list[-1]["extensions"] + 1
    df = pd.DataFrame.from_dict(log_list)
    return switches, df


def visualize_switches(data, switches):
    fig, ax = plt.subplots(1, 1)
    ax.plot(data)
    for x in switches:
        plt.axvline(x=x, color="red")
    plt.show()

def cluster_criterion(cluster_loss, window_loss, concatenation_loss):
    #todo: move this to a file, make this adaptable and provide different options
    return concatenation_loss < 10*cluster_loss and concatenation_loss < 10*window_loss

def cluster_segments(segments, data_frame, config):
    #todo: think about using previous models as starting point
    cluster_win = []

    for segment in segments:
        window = segment["window"]
        df_window = data_frame.slice(window[0], (window[1] - window[0]))

        learner = PySRRegressor(**config.get("kwargs", {}))
        learner.feature_names = config["features"]
        if not cluster_win:
            cluster_win.append([window])
            cluster_data = [df_window]
            cluster_loss = [[segment[config["selection"]]]]
            X_train = window[config["features"]]
            y_train = window[config["target_var"]]
            learner.fit(X_train, y_train)
            cluster_eq = [learner.sympy()]
            continue
        else:
            found_cluster = False
            for i, data in enumerate(cluster_data):
                concatenation = pl.concat([data, df_window])
                X_train = concatenation[config["features"]]
                y_train = concatenation[config["target_var"]]
                learner.fit(X_train, y_train)
                eq = learner.sympy()
                loss = learner.get_best()[config["selection"]]
                if cluster_criterion(mean(cluster_loss[i]),segment[config["selection"]],loss):
                    #todo: export to update cluster function
                    cluster_win[i].append(window)
                    cluster_data[i] = pl.concat([data,df_window])
                    cluster_loss[i].append(segment[config["selection"]])
                    cluster_eq[i] = eq
                    found_cluster = True
                    break
            
            if not found_cluster:
                cluster_win.append([window])
                cluster_data.append(df_window)
                cluster_loss.append([segment[config["selection"]]])
                X_train = window[config["features"]]
                y_train = window[config["target_var"]]
                learner.fit(X_train, y_train)
                cluster_eq.append(learner.sympy())
    
    print(cluster_win)
    print(cluster_eq)
    # todo: if neighbouring are one dynamic: combine them to one window?


def main(path):
    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(
        config["file"], dtypes=[pl.Float64] * len(config["features"])
    )
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]  # how to handle first time step? -> weight to 0?
        config["target_var"] = "diff"

    # Segmentation
    switches, results = identify_switch(config, data_frame)
    results.to_csv("segmentation_results.csv")
    print(switches)
    visualize_switches(data_frame[config["target_var"]], switches)

    # Clustering
    # optional: read previous results from file:
    #results = pl.read_csv("segmentation_results.csv")
    cluster_segments(results, data_frame, config)


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
