"""
FRE model from: Montbrió E, Pazó D and Roxin A 2015 Physical Review X 5 021028
solves:
dr/dt = Delta/Pi  + 2rv
dv/dt = v^2 - (pi*r)^2 + eta + J*r

where:
- r is a vector of length Npop
- v is a vector of length Npop

For the parameters:
- Delta is either a real number or vector of length Npop
- eta is either a real number or vector of length Npop
- J is a Npop x Npop matrix
- Iext is a function R -> R or R -> R^Npop and defines a time dependent external current
"""

import numpy as np


# default time dependent current of 0 for all times
def INone(t):
    return 0


class FRE_parameters:

    def __init__(self):
        self.t0 = 0
        self.Iext = INone

        self.tmax = None
        self.x0: np.ndarray = None

        self.Npop: int = None
        self.taum = None
        self.Delta = None
        self.eta = None
        self.J = None

        # tolerances for the solver
        self.rtol = 1e-7
        self.atol = 1e-12


class FRE_system:
    def __init__(self, parameters: FRE_parameters):
        self.p = parameters
        self.dxdt = np.zeros(parameters.x0.shape)

    def RHS(self, t, x: np.ndarray):
        p = self.p
        rs = x[0, :]
        vs = x[1, :]

        coupling = p.sigma * p.J.dot(rs)

        self.dxdt[0] = p.Delta / np.pi + 2 * rs * vs
        self.dxdt[1] = vs ** 2 + p.eta + p.Iext(t) - np.pi ** 2 * rs ** 2 + coupling

        return self.dxdt
