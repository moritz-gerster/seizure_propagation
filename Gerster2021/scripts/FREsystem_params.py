from parameters import parameters
import numpy as np


def INone(t):
    return 0


class FRE_parameters(parameters):
    N = None
    taum = None
    delta = None
    eta = None
    J = None
    Iext = None

    def __init__(self):

        parameters.__init__(self)
        self.Iext = INone


class FRE_1pop:

    parameters= None
    dxdt = None

    def __init__(self, params):

        self.parameters = params
        self.dxdt = np.zeros(params.x0.shape)

    def RHS(self, t,x):

        p = self.parameters
        self.dxdt[0] = p.delta / np.pi + 2*x[0] * x[1]
        self.dxdt[1] = x[1]**2 + p.eta + p.J*x[0] + p.Iext(t) - np.pi**2 * x[0]**2

        return self.dxdt


class FRE_multi_pop:

    parameters = None
    dxdt = None

    def __init__(self, params):

        self.parameters=params
        self.dxdt = np.zeros(params.x0.shape)

    def RHS(self, t,x):

        p = self.parameters
        rs = x[0, :]
        vs = x[1, :]

        coupling = p.sigma * p.J.dot(rs)

        self.dxdt[0] = p.delta / np.pi + 2*rs*vs
        self.dxdt[1] = vs**2 + p.eta + p.Iext(t) - np.pi**2 * rs**2 + coupling

        return self.dxdt
