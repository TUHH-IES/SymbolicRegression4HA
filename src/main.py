from gplearn.genetic import SymbolicRegressor

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def main() -> None:
    # Ground truth
    x0 = np.arange(-1, 1, .1)
    x1 = np.arange(-1, 1, .1)
    x0, x1 = np.meshgrid(x0, x1)
    #y_truth = x0**2 - x1**2 + x1 - 1

    data_frame = pl.read_csv("examples/data_nonoise.csv")

    window = 370
    # Training samples
    X_train = data_frame[['mQp','mUb','y1','Uo','h1','mUp','mQ0','vol1','vol2']].head(window) #'my1','my',
    y_train = data_frame['y2'].head(window)
    y_train = np.diff(y_train)
    y_train = np.insert(y_train, 0, y_train[0])

    est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, #number of generations for the symbolic regression learner
                           stopping_criteria=0.00000001, #
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=1.0, verbose=1,
                           parsimony_coefficient=0, #gives the trade-off between length of the equation and the fitting value (0 = no restriction to length) 
                           random_state=0,
                           function_set=('sub', 'mul', 'div','abs','sqrt'))
    #todo: implement sgn
    est_gp.fit(X_train, y_train)

    print(est_gp._program)

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