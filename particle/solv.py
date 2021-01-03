import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


class ChargeParticle(object):

    def __init__(self, idx=0):
        super().__init__()
        self.init_method(idx)
        self.solver = ode(self.newton).set_integrator('dopri5')

    def init_method(self, idx=0):
        if idx == 0:
            self.newton = self.newton0
        elif idx == 1:
            self.newton = self.newton1
        elif idx == 2:
            self.newton = self.newton2
        else:
            self.newton = self.newton0

    def e_of_x(self, x):
        return 10 * np.sign(np.sin(2 * np.pi * x / 25))

    def b_xyz(self, x=1, y=1, z=1):
        return 2 * x + np.sin(2 * np.pi * y / 25) + np.sin(2 * np.pi * z / 25)

    def newton0(self, t, Y, q, m):
        """
        Computes the derivative of the state vector y according to the equation of motion:
        Y is the state vector (x, y, z, u, v, w) === (position, velocity).
        returns dY/dt.
        """
        x, y, z = Y[0], Y[1], Y[2]
        u, v, w = Y[3], Y[4], Y[5]

        alpha = q / m * self.b_xyz(x, y, z)
        return np.array([u, v, w, 0.5, alpha * w + self.e_of_x(x), -alpha * v])

    def newton1(self, t, Y, q, m, B):
        """
        Computes the derivative of the state vector y according to the equation of motion:
        Y is the state vector (x, y, z, u, v, w) === (position, velocity).
        returns dY/dt.
        """
        x, y, z = Y[0], Y[1], Y[2]
        u, v, w = Y[3], Y[4], Y[5]

        alpha = q / m * B
        return np.array([u, v, w, 0, alpha * w, -alpha * v])

    def newton2(self, t, Y, q, m, B, E):
        """
        Computes the derivative of the state vector y according to the equation of motion:
        Y is the state vector (x, y, z, u, v, w) === (position, velocity).
        returns dY/dt.
        """
        x, y, z = Y[0], Y[1], Y[2]
        u, v, w = Y[3], Y[4], Y[5]

        alpha = q / m
        return np.array([u, v, w, 0, alpha * B * w + E, -alpha * B * v])
