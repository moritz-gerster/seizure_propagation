import numpy as np
from scipy.integrate import ode


class dp5_solver:
    
    rtol = None
    atol = None
    
    max_step = None

    dynamic_system = None
    parameters = None

    observer_list = None

    def __init__(self, dynamic_system):
        
        self.rtol = 1e-7
        self.atol = 1e-12
        
        
        
        self.dynamic_system = dynamic_system
        self.parameters = dynamic_system.parameters

        self.observer_list = []

    def integrate(self):
        p = self.parameters
        x0_shape = p.x0.shape
        

        if (np.ndim(p.x0) == 1):
            x0 = p.x0.copy()
            rhs = self.dynamic_system.RHS
            dp5_callback = self.observer_callback
        else:
            prod_shape = np.prod(x0_shape)
            x0 = p.x0.reshape(prod_shape)

            def rhs(t, x):
                return self.dynamic_system.RHS(t, x.reshape(x0_shape)).reshape(prod_shape)

            def dp5_callback(t,x):
                return self.observer_callback(t, x.reshape(x0_shape))


        solver = ode(rhs)

        max_step = 1
        if (self.max_step is not None):
            max_step = self.max_step


        solver.set_integrator('dopri5', nsteps=5e9, rtol=self.rtol, atol=self.atol, verbosity=0,
                              max_step=max_step)
        solver.set_solout(dp5_callback)

        solver.set_initial_value(x0, p.t0)
        solver.integrate(p.tmax)

        return solver

    def observer_callback(self, t, x):
        for obs in self.observer_list:
            obs.observe(t, x)
