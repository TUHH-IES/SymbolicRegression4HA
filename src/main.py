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
    'sign': lambda x: -1 if x < 0 else 1
}

def main() -> None:
    # Ground truth
    x0 = np.arange(-1, 1, .1)
    x1 = np.arange(-1, 1, .1)
    x0, x1 = np.meshgrid(x0, x1)
    #y_truth = x0**2 - x1**2 + x1 - 1

    data_frame = pl.read_csv("examples/data_nonoise.csv")

    window = 370
    # Training samples
    X_train = data_frame[['mQp','y1','y2','Uo','h1','mQ0','mUb']].head(window) #'my1','my','vol1','vol2','mUb','mUp'= mAp saturated
    # can I apply a normalization? -> this might not be conform with all of the operators...
    y_train = data_frame['y1'].head(window)
    y_train = np.diff(y_train)
    y_train = np.insert(y_train, 0, y_train[0])
    #y_train = (y_train-np.min(y_train))/(np.max(y_train)-np.min(y_train)) #normalize

    #additional functions:
    mydiv = make_function(function=functionals._protected_division,
                        name='mydiv',
                        arity=2)
    sign = make_function(function=functionals._sign,
                        name='sign',
                        arity=1)

    est_gp = SymbolicRegressor(population_size=5000,
                           generations=10, #number of generations for the symbolic regression learner
                           stopping_criteria=1e-5, #change this to improve fitness, adapt to assumed value range
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=1.0, verbose=1,
                           parsimony_coefficient= 1e-10, #"auto", 
                           #parsimony_coefficient=0, #gives the trade-off between length of the equation and the fitting value (0 = no restriction to length) 
                           random_state=0,
                           function_set=('sub','add','mul','abs','sqrt',mydiv,sign), #'div' might be crucial
                           feature_names=['mQp','y1','y2','Uo','h1','mQ0','mUb']
                           ) #'div', but that might not work good
    est_gp.fit(X_train, y_train)
    #final fitness (option: print score on training data)
    print(est_gp._program.raw_fitness_)

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

if __name__ == "__main__":
    main()