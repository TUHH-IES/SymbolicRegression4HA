from gplearn import fitness
import numpy as np

def _mape(y, y_pred, w):
    """Calculate the mean absolute percentage error."""
    diffs = np.abs(np.divide((np.maximum(0.001, y) - np.maximum(0.001, y_pred)),
                             np.maximum(0.001, y)))
    return 100. * np.average(diffs, weights=w)

mape = fitness.make_fitness(function=_mape,
                    greater_is_better=False)