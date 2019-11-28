# https://flothesof.github.io/charged-particle-trajectories-E-and-B-fields.html
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cnt
import sys
from scipy.integrate import ode

from OCC.Coregp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Coregp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.CoreTColgp import TColgp_Array1OfPnt
from OCCUtils.Construct import make_polygon

from base import plotocc, plot2d


class Particle (plotocc):

    def __init__(self, x0=gp_Pnt(0, 0, 1), v0=gp_Vec(0, 0, 10)):
        plotocc.__init__(self)
        self.solver = ode(self.newton).set_integrator('dopri5')
        self.initial_conditions = self.set_beam(x0, v0)
        self.q = 1.0
        self.m = 1.0
        self.pts = []

    def set_beam(self, p=gp_Pnt(), v=gp_Vec(0, 0, 1)):
        return [v.X(), p.Y(), p.Z(), v.X(), v.Y(), v.Z()]

    def get_beam(self, y=[0, 0, 0, 0, 0, 0]):
        p = gp_Pnt(*y[:3])
        v = gp_Vec(*y[3:])
        return p, v

    def e_xyz(self, x=1, y=1, z=1):
        # return 10 * np.sign(np.sin(2 * np.pi * x / 25))
        return 10 / (np.abs(x) + 1)

    def b_xyz(self, x=1, y=1, z=1):
        # return 2 * x + np.sin(2 * np.pi * y / 25) + np.sin(2 * np.pi * z / 25)
        return (-2 * (z - 0.5)**2 + 10)

    def newton(self, t, Y):
        """
        Computes the derivative of the state vector y according to the equation of motion:
        Y is the state vector (x, y, z, u, v, w) === (position, velocity).
        returns dY/dt.
        """
        x, y, z = Y[0], Y[1], Y[2]
        u, v, w = Y[3], Y[4], Y[5]

        alpha = self.q / self.m * self.b_xyz(x, y, z)
        return np.array([u, v, w, 0, alpha * self.b_xyz(x, y, z) * w + self.e_xyz(x), -alpha * self.b_xyz(x, y, z) * v])

    def compute_trajectory(self, t0=0.0, t1=100):
        r = ode(self.newton).set_integrator('dopri5')
        r.set_initial_value(
            self.initial_conditions, t0
        )
        positions = []
        t1 = 10
        dt = 0.01
        while r.successful() and r.t < t1:
            txt = "{:03.2f} / {:03.2f} ".format(r.t, t1)
            txt += "| {:03.2f} {:03.2f} {:03.2f} ".format(*r.y[:3])
            txt += "| {:03.2f} {:03.2f} {:03.2f} ".format(*r.y[3:])
            sys.stdout.write("\r" + txt)
            sys.stdout.flush()
            #r.set_f_params(m, q, 1.0, self.e_of_x(r.y[0]))
            r.integrate(r.t + dt)
            self.pts.append(gp_Pnt(*r.y[:3]))

        print()
        poly = make_polygon(self.pts)
        self.display.DisplayShape(poly)
        for pnt in self.pts[::20]:
            self.display.DisplayShape(pnt)


if __name__ == '__main__':
    pnt = gp_Pnt(1, 0, 0)
    vec = gp_Vec(1, 0, 1)
    print(cnt.c, "m/s")
    print(cnt.e)
    print(cnt.g)
    print(cnt.m_e, cnt.m_n, cnt.m_p, cnt.m_u)

    obj = Particle(pnt, vec)

    px = np.linspace(0, 15, 100)
    py = np.linspace(0, 15, 100)
    pz = np.linspace(0, 15, 100)
    mesh = np.meshgrid(px, py)

    pl2 = plot2d()
    pl2.axs.plot(px, obj.e_xyz(x=px, y=0, z=0))
    pl2.fig.savefig("ex.png")

    pl2.new_fig()
    pl2.axs.set_aspect("auto")
    pl2.axs.plot(pz, obj.b_xyz(x=0, y=0, z=pz))
    pl2.fig.savefig("bz.png")

    obj.compute_trajectory()
    obj.show_axs_pln(scale=10)
    obj.show()
