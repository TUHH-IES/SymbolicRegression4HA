import argparse
from ruamel.yaml import YAML
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import copy
import ast
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
    df = pl.DataFrame.from_dict(log_list)
    return switches, df


def visualize_switches(data, switches):
    fig, ax = plt.subplots(1, 1)
    ax.plot(data)
    for x in switches:
        plt.axvline(x=x, color="red")
    plt.show()


def cluster_criterion(cluster_loss, window_loss, concatenation_loss, factor):
    # todo: move this to a file, make this adaptable and provide different options -> only cluster loss, weighted mean of the two others?
    print("Cluster criterion", concatenation_loss, cluster_loss, window_loss)
    return (
        concatenation_loss < factor * cluster_loss #and concatenation_loss < factor * window_loss
    )


def cluster_segments(segments, data_frame, config):
    # todo: use previous models as starting point (option: 
        #1) try previous models for both. If one of them is better than the two before, a new one is found,
        #2) test fit first, then re-learn?)
    cluster_win = []
    fig, ax = plt.subplots(1, 1)
    ax.plot(data_frame["t"],data_frame[config["target_var"]],label = "gt")

    for segment in segments.iter_rows(named=True):
        window = segment["window"]
        print("Current window:", window)
        df_window = data_frame.slice(window[0], (window[1] - window[0]))

        learner = PySRRegressor(**config.get("kwargs", {}))
        learner.warm_start = False
        learner.feature_names = config["features"]
        if not cluster_win:
            cluster_win.append([window])
            cluster_data = [df_window]
            cluster_loss = [[segment[config["selection"]]]]
            X_train = df_window[config["features"]]
            y_train = df_window[config["target_var"]]
            learner.fit(X_train, y_train)
            cluster_eq = [learner.sympy()]
            ax.plot(df_window["t"],learner.predict(X_train),label = "window" + ",".join(str(element) for element in window) + "newcluster")
            continue
        else:
            found_cluster = False
            for i, data in enumerate(cluster_data):
                print("Current cluster",i,"of",len(cluster_win))
                learner.equation_file = (
                "./equations/"
                + config["file_prefix"]
                + "_win"
                + str(window[1])
                + "_cluster"
                + str(i)
                + ".csv"
                )
                concatenation = pl.concat([data, df_window])
                X_train = concatenation[config["features"]]
                y_train = concatenation[config["target_var"]]
                learner.fit(X_train, y_train)
                eq = learner.sympy()
                loss = learner.get_best()[config["selection"]]
                if cluster_criterion(
                    mean(cluster_loss[i]), segment[config["selection"]], loss, config["cluster_criterion"]["factor"]
                ):
                    # todo: export to update cluster function
                    print("cluster", window, "into", i)
                    cluster_win[i].append(window)
                    cluster_data[i] = pl.concat([data, df_window])
                    cluster_loss[i].append(segment[config["selection"]])
                    cluster_eq[i] = eq
                    found_cluster = True
                    ax.plot(concatenation["t"],learner.predict(X_train),label = "window" + ",".join(str(element) for element in window) + "cluster" + str(i))
                    break
                #alternative procedure: test against all clusters and choose the smallest one, if the error is below something or the increase in accuracy is large enough

            if not found_cluster:
                print("Create new cluster")
                cluster_win.append([window])
                cluster_data.append(df_window)
                cluster_loss.append([segment[config["selection"]]])
                X_train = df_window[config["features"]]
                y_train = df_window[config["target_var"]]
                learner.fit(X_train, y_train)
                cluster_eq.append(learner.sympy())
                ax.plot(df_window["t"],learner.predict(X_train),label = "window" + ",".join(str(element) for element in window) + "newcluster")

    print(cluster_win)
    print(cluster_eq)
    ax.legend()
    plt.show()
    return cluster_win, cluster_eq
    # todo: if neighbouring are one dynamic: combine them to one window?

def visualize_cluster(data, clusters):
    cmap = plt.colormaps.get_cmap('hsv')
    fig, ax = plt.subplots(1, 1)
    ax.plot(data)
    for i, cluster in enumerate(clusters):
        alpha = 0.8 - i / len(clusters)
        for window in cluster:
            plt.axvspan(window[0], window[1], color=cmap(i/len(clusters)),alpha = alpha)
    plt.show()

def main(path):
    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(config["file"], dtypes=[pl.Float64] * len(config["features"]))
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][
            1
        ]  # how to handle first time step? -> weight to 0?
        config["target_var"] = "diff"

    if "selection" not in config:
        config["selection"] = "loss"

    # Segmentation
    #switches, results = identify_switch(config, data_frame)
    #results.to_csv("segmentation_results.csv")
    #print(switches)
    #visualize_switches(data_frame[config["target_var"]], switches)

    # Clustering
    # optional: read previous results from file:
    pandas_results = pd.read_csv("segmentation_results.csv")
    pandas_results['window'] = pandas_results['window'].apply(lambda x: ast.literal_eval(x))
    results = pl.DataFrame(pandas_results)

    cluster, equations = cluster_segments(results, data_frame, config)
    visualize_cluster(data_frame[config["target_var"]], cluster)
    cluster = pd.DataFrame.from_dict(cluster)
    equations = pd.DataFrame.from_dict(equations)
    cluster.to_csv("cluster.csv")
    equations.to_csv("equations.csv")



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
