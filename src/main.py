from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sympy import sympify, simplify
import math

import functionals

converter = {
    'sub': lambda x, y: x - y,
    'div': lambda x, y: x / y,
    'mydiv': lambda x, y: x / y,
    'mul': lambda x, y: x * y,
    'add': lambda x, y: x + y,
    'neg': lambda x: -x,
    'pow': lambda x, y: x ** y,
    'sin': lambda x: math.sin(x),
    'cos': lambda x: math.cos(x),
    'inv': lambda x: 1 / x,
    'inv_custom': lambda x: 1 / x,
    'sqrt': lambda x: x ** 0.5,
    'sqrt_custom': lambda x: x ** 0.5,
    'pow3': lambda x: x ** 3,
    'abs': lambda x: abs(x),
    #'sign': lambda x: -1 if x < 0 else 1 cannot use conditions here
}

def calculate(mode) -> None:

    data_frame = pl.read_csv("examples/data_nonoise.csv")

    window = 370
    data_frame = data_frame.head(window) #Features: 'mQp','y1','y2','Uo','h1','mQ0','mUb', 'my1','my','vol1','vol2','mUb','mUp'= mAp saturated

    #additional functions:
    mydiv = make_function(function=functionals._protected_division,
                        name='mydiv',
                        arity=2)
    sign = make_function(function=functionals._sign,
                        name='sign',
                        arity=1)

    est_gp = SymbolicRegressor(population_size=5000,
                           p_crossover=0.5, p_subtree_mutation=0.2,
                           p_hoist_mutation=0.05, p_point_mutation=0.2,
                           max_samples=1.0, verbose=1,
                           #parsimony_coefficient=0, #gives the trade-off between length of the equation and the fitting value (0 = no restriction to length) 
                           random_state=0,
                           function_set=('sub','add','mul','abs','sqrt',mydiv,sign), #'div' might be crucial
                           ) #'div', but that might not work good
    
    if mode == "standard-y1":
        feature_names = ['mQp','y1','y2','h1','mUb']
        est_gp.feature_names = feature_names
        est_gp.stopping_criteria=1e-5
        est_gp.generations=10
        est_gp.parsimony_coefficient= 1e-5
        X_train = data_frame[feature_names]
        y_train = data_frame['y1']
        y_train = np.diff(y_train)
        y_train = np.insert(y_train, 0, y_train[0])
        est_gp.fit(X_train, y_train)
    elif mode == "subtract-mQp":
        feature_names = ['y1','y2','h1','mUb']
        est_gp.feature_names = feature_names
        est_gp.stopping_criteria=1e-5
        est_gp.generations=20
        est_gp.parsimony_coefficient= 1e-3
        X_train = data_frame[['y1','y2','h1','mUb']]

        y_train = data_frame['y1']
        y_train = np.diff(y_train)
        y_train = np.insert(y_train, 0, y_train[0])
        y_train = y_train - data_frame["mQp"]
        est_gp.fit(X_train, y_train)

    print(est_gp._program.raw_fitness_) #final fitness (option: print score on training data)
    #print(est_gp._program)
    label = f"{sympify(str(est_gp._program), locals=converter)}"
    label = simplify(label)
    print(label)

    y_gp = est_gp.predict(X_train)
    #score_gp = est_gp.score(X_test, y_test)

    fig, ax = plt.subplots(1, 1)
    ax.plot(y_train,label="gt")
    ax.plot(y_gp,label="gp")
    ax.legend()
    plt.show()
    #print(score_gp)

def plot_correct():

    data_frame = pl.read_csv("examples/data_nonoise.csv")

    window = 370
    data_frame = data_frame.head(window)
    Cvb = 1.5938*1e-4
    A1 = 0.0154
    calculated = (data_frame["mQp"] - Cvb * np.sign(data_frame["y1"] - data_frame["y2"])* np.sqrt(abs(data_frame["y1"]-data_frame["y2"]))*data_frame["mUb"]) / A1
    y_train = data_frame['y1']
    y_train = np.diff(y_train)
    y_train = np.insert(y_train, 0, y_train[0])
    y_train = y_train * (calculated[0] / y_train[0]) # coefficient needs to be added

    fig, ax = plt.subplots(1, 1)
    ax.plot(y_train,label="gt")
    ax.plot(calculated,label="calc")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    plot_correct()
    #calculate("subtract-mQp")
    

