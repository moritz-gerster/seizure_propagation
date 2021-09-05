"""Oberserver class."""
import numpy as np
from timeit import default_timer as timer


class observer_base:
    parameters = None

    def __init__(self, parameters):
        self.parameters = parameters

    def observe(self, t, x: np.ndarray):
        pass


class obs_orbit(observer_base):
    xv = None
    tv = None

    def __init__(self, parameters):
        observer_base.__init__(self, parameters)
        self.xv = []
        self.tv = []

    def observe(self, t, x):
        self.tv.append(t)
        self.xv.append(x.copy())


class obs_output(observer_base):

    def __init__(self, parameters, output_delay=0.5):
        observer_base.__init__(self, parameters)
        self.T = timer()
        self.output_delay = output_delay

    def observe(self, t, x):
        p = self.parameters
        current_time = timer()
        time_diff = current_time - self.T
        if (time_diff > self.output_delay or t == p.t0 or t == p.tmax ):
            self.T = current_time
            print(f"{time_diff:.2f} s elapsed | Simulation time {t:.2f}/{p.tmax}")
