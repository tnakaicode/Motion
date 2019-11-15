# https://ipython-books.github.io/123-simulating-an-ordinary-differential-equation-with-scipy/
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cnt
import sys
from scipy.integrate import ode

from OCC.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.gp import gp_Pln
from OCC.IntAna import IntAna_IntConicQuad, IntAna_Quadric
from OCC.IntCurve import IntCurve_IConicTool
from OCC.IntTools import IntTools_Curve
from OCC.BRep import BRep_Tool
from OCC.Geom import Geom_Curve, Geom_Line, Geom_Curve
from OCC.GeomAPI import GeomAPI_IntCS
from OCC.GeomInt import GeomInt_LineTool
from OCC.GeomTools import GeomTools_CurveSet
from OCC.GeomLProp import GeomLProp_SurfaceTool, GeomLProp_CurveTool
from OCC.TColgp import TColgp_Array1OfPnt
from OCCUtils.Construct import make_polygon
from OCCUtils.Construct import make_edge, make_plane

from base import plotocc, plot2d
from base import pln_for_axs


class Particle (plotocc):

    def __init__(self, x0=[-10, 0, 10], v0=[1000, 5, 1000]):
        plotocc.__init__(self)
        self.solver = ode(self.newton).set_integrator('dopri5')
        self.initial_conditions = np.concatenate((x0, v0))
        self.m = 10.0
        self.k = 0.0

        self.grd_axs = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(-0.25, 0, 1))
        self.grd = pln_for_axs(self.grd_axs, [500, 500])
        self.pts, self.vel = [], []

    def newton(self, t, Y):
        """
        Computes the derivative of the state vector y according to the equation of motion:
        Y is the state vector (x, y, z, u, v, w) === (position, velocity).
        returns dY/dt.
        """
        px, py, pz = Y[0], Y[1], Y[2]
        vx, vy, vz = Y[3], Y[4], Y[5]

        alpha = - self.k / self.m
        ux, uy, uz = alpha * vx, alpha * vy, alpha * vz
        uy += -0.5 * py
        uz -= cnt.g / (pz)
        return np.array([vx, vy, vz, ux, uy, uz])

    def compute_trajectory(self, t0=0.0, t1=2000):
        r = ode(self.newton).set_integrator('dopri5')
        r.set_initial_value(
            self.initial_conditions, t0
        )
        positions = []
        dt = 1.0
        self.pts, self.vel = [], []
        while r.successful() and r.t < t1:
            txt = "{:03.2f} / {:03.2f} ".format(r.t, t1)
            txt += "| {:03.2f} {:03.2f} {:03.2f} ".format(*r.y[:3])
            txt += "| {:03.2f} {:03.2f} {:03.2f} ".format(*r.y[3:])
            sys.stdout.write("\r" + txt)
            sys.stdout.flush()
            self.pts.append(gp_Pnt(*r.y[:3]))
            self.vel.append(gp_Vec(*r.y[3:]))
            # self.check_ground()
            r.integrate(r.t + dt)

        print()
        poly = make_polygon(self.pts)
        self.display.DisplayShape(poly)
        for pnt in self.pts[::10]:
            self.display.DisplayShape(pnt)
        self.display.DisplayShape(self.grd, transparency=0.8, color="BLUE")

    def check_ground(self):
        if len(self.pts) > 2:
            ray = make_edge(self.pts[-2], self.pts[-1])
            h_line = BRep_Tool.Curve(ray)
            h_surf = BRep_Tool.Surface(self.grd)
            print(GeomAPI_IntCS(h_line, h_surf))
            #print(GeomAPI_IntCS(h_line, h_surf).IsDone())


if __name__ == '__main__':
    pnt = gp_Pnt(0, 0, 1)
    vec = gp_Vec(0, 0, 10)
    print(cnt.c, "m/s")
    print(cnt.e)
    print(cnt.g)
    print(cnt.m_e, cnt.m_n, cnt.m_p, cnt.m_u)

    obj = Particle()
    obj.compute_trajectory()
    obj.show_axs_pln(scale=1000000)
    obj.show()
