import argparse
from ruamel.yaml import YAML
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from sympy import sympify, simplify

from pysr import PySRRegressor

import functionals

def identify_switch(path):
    config = YAML(typ="safe").load(path)

    data_frame = pl.read_csv(config["file"])

    #todo: make these parameters part of the config or calculate from data length?
    start_width = 70
    step_width = 20
    window = [0,start_width] #running index to capture the current window of data considered

    #set up regressor
    learner = PySRRegressor(**config.get("kwargs", {}))
    learner.feature_names = config["features"]

    fitness = 100 #option: think about using best fitness seen here
    new_fitness = 50
    switches = [0]
    while window[1] < len(data_frame):
        learner.warm_start = False
        while new_fitness < fitness and window[1] < len(data_frame): #if fitness gets worse (larger) by a specific degree; option: if fitness gets worse at all
            print(window)
            current_frame = data_frame.slice(window[0],window[1]) #todo: do I have to give all or for warm start just those that are new?
            #option: think about giving more weight to new data points
            fitness = new_fitness
            #Assume for the moment that target is observed
            X_train = current_frame[config["features"]]
            y_train = current_frame[config["target_var"]]
            learner.fit(X_train, y_train)
            new_fitness = learner._program.raw_fitness_
            window[1] += step_width
            learner.warm_start = True
            learner.generations += 3 # make this a parameter
        #for output:
        label = f"{sympify(str(learner._program), locals=functionals.converter)}"
        label = simplify(label)
        print(label)
        switches.append(window[1])
        window[0] = window[1] - step_width 
        window[1] = window[0] + start_width
        #reset fitness
        fitness = 100 
        new_fitness = 50
        learner.generations = config["kwargs"]["generations"]

    print(switches)
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
    

