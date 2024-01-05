import polars as pl
import numpy as np
import matplotlib.pyplot as plt

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
    #print(calculated[0] / y_train[0] / A1)
    y_train = y_train * (calculated[0] / y_train[0]) # coefficient needs to be added --> Why?

    fig, ax = plt.subplots(1, 1)
    ax.plot(y_train,label="gt")
    ax.plot(calculated,label="calc")
    ax.legend()
    plt.show()