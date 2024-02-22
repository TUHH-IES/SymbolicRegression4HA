import numpy as np

def average_increase(fitness_hist, saturation = 1e-10, factor = 1):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] > saturation or factor * fitness_list[-1] >= np.mean(fitness_list[0:-2])

def average_decrease(fitness_hist, saturation = 1e-10, factor = 1):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] < saturation or factor * fitness_list[-1] <= np.mean(fitness_list[0:-2])

def increase(fitness_hist, saturation = 1e-10, factor = 1):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] > saturation or factor * fitness_list[-1] >= fitness_list[-2]

def decrease(fitness_hist, saturation = 1e-10, factor = 1):
    fitness_list = [*fitness_hist]
    return fitness_list[-1] < saturation or factor * fitness_list[-1] <= fitness_list[-2]