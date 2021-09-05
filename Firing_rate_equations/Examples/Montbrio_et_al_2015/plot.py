import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_solution(df: pd.DataFrame, show=True):
    fig, ax = plt.subplots(2, 2, figsize=(10, 4))

    for sub_ax in ax[-1]:
        sub_ax.set_xlabel("Time $t$")

    ax[0, 0].set_ylabel("$r(t)$")
    ax[1, 0].set_ylabel("$v(t)$")

    for sub_ax in ax[:, 0]:
        sub_ax.set_xlim(-10, 40)

    for sub_ax in ax[:, 1]:
        sub_ax.set_xlim(-10, 80)

    tv = df.attrs["tv"]
    xv = df.attrs["xv"]
    p = df.attrs["p"]

    print(p)
    print(xv[0, 0])

    for k in range(2):
        for j in range(2):
            ax[k, j].plot(tv, xv[:, k, j])

    plt.tight_layout()
    if (show):
        plt.show()

    return fig, ax


if (__name__ == "__main__"):
    df = pd.read_pickle("data/result.pkl")

    plot_solution(df)
