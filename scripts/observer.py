"""Oberserver class."""
import numpy as np
from timeit import default_timer as timer


class observer:
    parameters = None

    def __init__(self, parameters):
        self.parameters = parameters


class obs_time(observer):

    tv = None

    def __init__(self, parameters):
        observer.__init__(self, parameters)
        self.tv = []

    def observe(self, t, x):

        self.tv.append(t)


class obs_traj(observer):

    xv = None

    def __init__(self, parameters):
        observer.__init__(self, parameters)
        self.xv = []

    def observe(self, t, x):

        self.xv.append(np.copy(x))


class obs_order_param(observer):

    ov = None

    def __init__(self, parameters):
        observer.__init__(self, parameters)
        self.ov = []

    def observe(self, t, x):

        p = self.parameters
        order_parameter = np.sum(np.exp(1j*x)) / p.N
        self.ov.append(order_parameter)


class obs_output(observer):

    T = None

    def __init__(self, parameters):
        observer.__init__(self, parameters)
        self.T = timer()

    def observe(self, t, x):
        current_time = timer()
        time_diff = current_time - self.T
        if time_diff > 1:
            self.T = current_time
