"""
Simulation program base class:
- handles the dynamical system to be solved, in this case FRE_model.FRE_system()
- handles the observer for the orbit and terminal output
- saves panda dataframe

We can interhit from this class to define parameter values and run more specific simulations, see examples
"""

import numpy as np
import observer
import solver
import FRE_model
import pandas as pd


class program_base:

    def __init__(self):
        self.p = FRE_model.FRE_parameters()
        self.df: pd.DataFrame = pd.DataFrame()

    def run_simulation(self, save_path=None):
        print("Running simulation with following parameters:")
        print(self.p.__dict__)
        print("=" * 100)
        dyn_system = FRE_model.FRE_system(self.p)
        s = solver.dp5_solver(dyn_system)

        obs_orbit = observer.obs_orbit(self.p)
        obs_output = observer.obs_output(self.p)

        s.observer_list.append(obs_orbit)
        s.observer_list.append(obs_output)

        s.integrate()

        self.df.attrs["tv"] = np.array(obs_orbit.tv)
        self.df.attrs["xv"] = np.array(obs_orbit.xv)
        self.df.attrs["p"] = self.p

        print("=" * 100)
        print("Done.")

        if (save_path is not None):
            self.save_dataframe(save_path)

    def save_dataframe(self, path):
        Iext_tmp = self.p.Iext
        self.p.Iext = None  # necessary to be able to load the parameters

        self.df.to_pickle(path)

        self.p.Iext = Iext_tmp
