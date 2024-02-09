import numpy as np

def average_increase(fitness_hist):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] >= np.mean(fitness_list[0:-2])

def strict_increase(fitness_hist):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] >= fitness_list[-2]

def strict_decrease(fitness_hist):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] <= fitness_list[-2]

def strict_decrease_bounded(fitness_hist):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] < 1e-10 or fitness_list[-1] <= fitness_list[-2]