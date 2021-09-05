"""
This example reproduces the FRE equation results from:
Montbrió E, Pazó D and Roxin A 2015 Physical Review X 5 021028
- Figure 2
"""

import sys

sys.path.append('../..') # not elegant workaround to easily import parent directory modules

import numpy as np
from FRE_program import program_base

import plot

# just an empty 2d array to be copied when calculating the external current, improves performance
current_dummy = np.zeros(2)


# stimulation current for the two cases: step and sinusoidal
def stim(t):
    ret = current_dummy.copy()
    if (0 < t < 30):
        ret[0] = 3
    if (0 < t):
        omega = np.pi / 20
        ret[1] = 3 * np.sin(omega * t)

    return ret


class subprogram(program_base):

    def __init__(self):
        program_base.__init__(self)

        p = self.p

        p.Npop = 2
        # We use two uncoupled populations to simulate both columns of the figure at the same time
        # pop. 1 receives step current
        # pop. 2 receives sinusoidal current

        p.t0 = -40
        p.tmax = 80
        p.x0 = np.zeros((2, p.Npop))

        p.sigma = 1
        p.J = np.zeros((p.Npop, p.Npop))

        p.J[0, 0] = 15
        p.J[1, 1] = 15
        p.Delta = 1
        p.eta = -5

        p.Iext = stim


if (__name__ == "__main__"):
    prog = subprogram()
    prog.run_simulation(save_path="data/result.pkl")

    plot.plot_solution(prog.df)
