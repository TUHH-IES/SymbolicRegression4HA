from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    # Ground truth
    x0 = np.arange(-1, 1, .1)
    x1 = np.arange(-1, 1, .1)
    x0, x1 = np.meshgrid(x0, x1)
    #y_truth = x0**2 - x1**2 + x1 - 1

    rng = check_random_state(0)

    # Training samples
    X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1

    # Testing samples
    X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1

    est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train, y_train)

    print(est_gp._program)

    y_gp = est_gp.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    score_gp = est_gp.score(X_test, y_test)

    fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

    axs.plot_surface(x0, x1, y_gp, rstride=1, cstride=1, color='green', alpha=0.5)
    axs.scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.show()
    print(score_gp)

if __name__ == "__main__":
    main()