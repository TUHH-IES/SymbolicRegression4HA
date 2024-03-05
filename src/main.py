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

from pysr import PySRRegressor

import criteria

def identify_switch(path):
    config = YAML(typ="safe").load(path)

    data_frame = pl.read_csv(config["file"], dtypes = [pl.Float64] * len(config["features"]))

    #todo: add an 'auto'-option which estimates this from data
    start_width = config["start_width"]
    step_width = config["step_width"]
    step_iterations = config["step_iterations"]
    hist_length = config["hist_length"]
    criterion = getattr(criteria, config["criterion"]["name"])
    if "kwargs" in config["criterion"]:
        criterion = partial(criterion,**config["criterion"]["kwargs"])
    if "selection" in config:
        selection = config["selection"]
    else:
        selection = "loss"
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1] #how to handle first time step?
        config["target_var"] = "diff"

    learner = PySRRegressor(**config.get("kwargs", {}))
    learner.feature_names = config["features"]

    fitness_hist = deque([],hist_length)
    switches = [0]
    window = [0,start_width-step_width]
    log_list = []
    while window[1] < len(data_frame):
        learner.warm_start = False
        fitness_hist = deque([],hist_length)
        extension = 0
        while len(fitness_hist) < 2 or (criterion(fitness_hist) and window[1] < len(data_frame)):
            learner.equation_file = "./equations/" + config["file_prefix"] + "_win" + str(len(switches)) + "_ext" + str(extension) + ".csv"
            if hasattr(learner,'equations_'): 
                best_equation = learner.sympy()
            window[1] += step_width
            window[1] = min(window[1],len(data_frame))

            print(window)
            current_frame = data_frame.slice(window[0],(window[1]-window[0]))
            
            X_train = current_frame[config["features"]]
            y_train = current_frame[config["target_var"]]
            learner.fit(X_train, y_train)
            fitness_hist.append(learner.get_best()[selection])
            learner.warm_start = True
            learner.niterations = step_iterations
            extension = extension + 1
            log_best = []
            log_best.append([window,learner.sympy(),learner.get_best()["loss"],learner.get_best()["score"]])
            df_log = pd.DataFrame.from_dict(log_best)
            df_log.to_csv("./equations/" + config["file_prefix"] + "_win" + str(len(switches)) + "_ext" + str(extension) + "_best.csv")

        log = dict()
        log["extensions"] = extension-1
        log["window"] = copy.deepcopy(window)
        log["window"][1] = log["window"][1] - step_width
        log["equation"] = best_equation
        log_list.append(log)
        switches.append(window[1] - step_width)
        window[0] = window[1] - step_width
        window[1] = min(window[0] + start_width - step_width, len(data_frame))
        learner.niterations = config["kwargs"]["niterations"]

    switches[-1] = len(data_frame)
    log_list[-1]["window"][1] = len(data_frame)
    log_list[-1]["extensions"] = log_list[-1]["extensions"]+1
    print(switches)
    df = pd.DataFrame.from_dict(log_list)
    df.to_csv("results.csv")
    fig, ax = plt.subplots(1, 1)
    ax.plot(data_frame[config["target_var"]])
    for x in switches:
        plt.axvline(x = x,color = "red")
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
    identify_switch(arguments.config)
    

