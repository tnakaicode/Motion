# https://flothesof.github.io/charged-particle-trajectories-E-and-B-fields.html
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

from particle.base import plot3d, plotocc
from particle.solv import ChargeParticle


class Particle (plot3d):

    def __init__(self):
        plot3d.__init__(self)
        self.sol = ChargeParticle()

    def compute_trajectory(self, m=1, q=1):
        r = ode(self.newton1).set_integrator('dopri5')
        r.set_initial_value(initial_conditions,
                            t0).set_f_params(m, q, 1.0, 10.)
        positions = []
        t1 = 200
        dt = 0.05
        while r.successful() and r.t < t1:
            r.set_f_params(m, q, 1.0, self.e_of_x(r.y[0]))
            r.integrate(r.t + dt)
            positions.append(r.y[:3])
        return np.array(positions)


if __name__ == '__main__':
    obj = Particle()

    t0 = 0
    x0 = np.array([0, 0, 0])
    v0 = np.array([1, 1, 0])
    initial_conditions = np.concatenate((x0, v0))

    obj.sol.solver.set_initial_value(initial_conditions, t0)
    obj.sol.solver.set_f_params(-1.0, 1.0)

    pos = []
    t1 = 11.0
    dt = 0.005
    while obj.sol.solver.successful() and obj.sol.solver.t < t1:
        obj.sol.solver.integrate(obj.sol.solver.t + dt)
        pos.append(obj.sol.solver.y[:3])

        txt = "\r {:02.2f} - ".format(obj.sol.solver.t)
        for v in obj.sol.solver.y:
            txt += "{:.3f}\t".format(v)
        sys.stdout.write(txt)
        sys.stdout.flush()
    pos = np.array(pos)

    obj.axs.plot3D(pos[:, 0], pos[:, 1], pos[:, 2])
    obj.SavePng()
    plt.show()
