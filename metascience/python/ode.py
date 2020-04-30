import numpy as np
from scipy.integrate import solve_ivp as solver
import pdb
import matplotlib.pyplot as plt

def generate_oscillator_timeseries(pars, t0, y0, interval = None, t_obs = None):

    # Cosmological parameters
    k = pars[0]
    m = pars[1]

    # Systematics parameters
    c = pars[2]
    A = pars[3]
    wd = pars[4]
    phid = pars[5]

    def forcing_function(t):
        return A*np.cos(wd*t+phid)

    def oscillator_eqns(t,y):
        x = y[0]
        u = y[1]
        xp = u
        up = (forcing_function(t) - c*u - k*x)/m
        return np.array([xp,up])


    solution = solver(oscillator_eqns,interval, y0, t_eval = t_obs)
    return solution

if __name__=='__main__':
    parameters = np.array([1000.0, 10.0, 10.0, 0.1, 0.25, 0.9])
    t_solution = np.linspace(0,1,50.)
    t_span = (np.min(t_solution), np.max(t_solution))
    t0 = 0.0
    y0 = np.array([0.0, 1.0])

    solution = generate_oscillator_timeseries(parameters, t0, y0,
        interval = t_span, t_obs = t_solution)
    plt.plot(solution.t, solution.y[0])
    plt.show()
    pdb.set_trace()
