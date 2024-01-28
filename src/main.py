import argparse
from ruamel.yaml import YAML
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sympy import sympify, simplify

from gplearn.genetic import SymbolicRegressor

import functionals
import fitness

def identify_switch(path):
    config = YAML(typ="safe").load(path)

    data_frame = pl.read_csv(config["file"])

    #todo: make these parameters part of the config or calculate from data length?
    start_width = 70
    step_width = 20
    window = [0,start_width] #running index to capture the current window of data considered

    threshold = 0.0005

    #set up regressor
    est_gp = SymbolicRegressor(**config.get("kwargs", {}))
    est_gp.feature_names = config["features"]
    function_set = tuple(config["function_set"])
    function_set = function_set + tuple([getattr(functionals, name) for name in config["additional_functions"]])
    est_gp.function_set = function_set
    if "custom_metric" in config:
        est_gp.metric = getattr(fitness,config["custom_metric"])

    fitness = 100 #option: think about using best fitness seen here
    new_fitness = 50
    switches = [0]
    while window[1] < len(data_frame):
        est_gp.warm_start = False
        while new_fitness < fitness and window[1] < len(data_frame): #if fitness gets worse (larger) by a specific degree; option: if fitness gets worse at all
            print(window)
            current_frame = data_frame.slice(window[0],window[1]) #todo: do I have to give all or for warm start just those that are new?
            #option: think about giving more weight to new data points
            fitness = new_fitness
            #Assume for the moment that target is observed
            X_train = current_frame[config["features"]]
            y_train = current_frame[config["target_var"]]
            est_gp.fit(X_train, y_train)
            new_fitness = est_gp._program.raw_fitness_
            window[1] += step_width
            est_gp.warm_start = True
            est_gp.generations += 3 # make this a parameter
        #for output:
        label = f"{sympify(str(est_gp._program), locals=functionals.converter)}"
        label = simplify(label)
        print(label)
        switches.append(window[1])
        window[0] = window[1] - step_width 
        window[1] = window[0] + start_width
        #reset fitness
        fitness = 100 
        new_fitness = 50
        est_gp.generations = config["kwargs"]["generations"]

    print(switches)
    fig, ax = plt.subplots(1, 1)
    ax.plot(data_frame[config["target_var"]])
    for x in switches:
        plt.axvline(x = x,color = "red")
    plt.show()

def calculate(path) -> None:

    config = YAML(typ="safe").load(path)

    data_frame = pl.read_csv(config["file"])

    window = config["window"]
    data_frame = data_frame.slice(window["start"],window["length"])

    est_gp = SymbolicRegressor(**config.get("kwargs", {}))
    est_gp.feature_names = config["features"]
    function_set = tuple(config["function_set"])
    function_set = function_set + tuple([getattr(functionals, name) for name in config["additional_functions"]])
    est_gp.function_set = function_set
    if "custom_metric" in config:
        est_gp.metric = getattr(fitness,config["custom_metric"])

    X_train = data_frame[config["features"]]
    
    if "target_manipulation" in config:
        if config["target_manipulation"] == "differentiate":
            y_train = data_frame[config["target_var"]]
            y_train = np.gradient(y_train)
        elif config["target_manipulation"]  == "two-tank-subtract-dominant":
            Cvb = 1.5938*1e-4
            y_train = 1000* (Cvb * np.sign(data_frame["y1"] - data_frame["y2"])* np.sqrt(abs(data_frame["y1"]-data_frame["y2"]))*data_frame["mUb"])
        else:
            y_train = data_frame[config["target_var"]]
    else:
        y_train = data_frame[config["target_var"]]

    #sample_weights = np.concatenate((10*np.ones(50),np.ones(270),10*np.ones(50)))
    est_gp.fit(X_train, y_train) #,sample_weight=sample_weights)

    print(est_gp._program.raw_fitness_) #final fitness (option: print score on training data)
    label = f"{sympify(str(est_gp._program), locals=functionals.converter)}"
    label = simplify(label)
    print(label)

    y_gp = est_gp.predict(X_train)

    fig, ax = plt.subplots(1, 1)
    ax.plot(y_train,label="gt")
    ax.plot(y_gp,label="gp")
    ax.legend()
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
    #calculate(arguments.config)
    identify_switch(arguments.config)
    

