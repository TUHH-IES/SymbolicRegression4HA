import argparse
from ruamel.yaml import YAML
from pathlib import Path
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sympy import sympify, simplify

from gplearn.genetic import SymbolicRegressor

import functionals
import fitness

def calculate(path) -> None:

    config = YAML(typ="safe").load(path)

    data_frame = pl.read_csv(config["file"],dtypes={"dy":pl.Float64,"sumsqrt":pl.Float64})

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
    dir, file_name = os.path.split(path)
    file_name = file_name.replace("yaml","png")
    plt.savefig(os.path.join(dir,file_name))
    file_name = file_name.replace("png","txt")

    with open(os.path.join(dir,file_name), 'a') as f:
        f.write(str(label))
        f.write("\n")
        f.write(json.dumps(est_gp.run_details_))

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
    

