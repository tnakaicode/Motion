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
from base import pln_for_axs, gen_ellipsoid, spl_2pnt

# Earth Radius: 6, 371 km
# F_g = G m_1 * m_2 / (r_1 * r_2)
# G := 6.70883 E-11 m^(-3)kg^(-1)s^(-2)
# g := G m_1 / r_1 = 6.6742E-11 * 5.9736E+34 / (6.37101E+6)^2


class Particle (plotocc):

    def __init__(self, pnt=gp_Pnt(-10, 0, -10), vec=gp_Vec(10, 20, 100)):
        plotocc.__init__(self)
        self.solver = ode(self.newton).set_integrator('dopri5')
        self.initial_conditions = self.gen_condition(pnt, vec)
        self.m = 10.0
        self.k = 0.0
        ray = spl_2pnt()

        self.grd_axs = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(-0.25, 0, 1))
        self.grd = pln_for_axs(self.grd_axs, [500, 500])
        self.h_surf = BRep_Tool.Surface(self.grd)
        self.p_trce = GeomAPI_IntCS(ray, self.h_surf)

        self.pts, self.vel = [], []

    def gen_condition(self, pnt=gp_Pnt(), vec=gp_Vec(0, 0, 1)):
        x, y, z = pnt.X(), pnt.Y(), pnt.Z()
        u, v, w = vec.X(), vec.Y(), vec.Z()
        return np.array([x, y, z, u, v, w])

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
        #uz -= cnt.g / (pz)
        uz -= cnt.g
        return np.array([vx, vy, vz, ux, uy, uz])

    def compute_trajectory(self, t0=0.0, t1=10.0):
        self.r = ode(self.newton).set_integrator('dopri5')
        self.r.set_initial_value(
            self.initial_conditions, t0
        )
        positions = []
        self.dt = 1.0
        self.pts, self.vel = [], []
        while self.r.successful() and self.r.t < t1:
            txt = "{:03.2f} / {:03.2f} ".format(self.r.t, t1)
            txt += "| {:03.2f} {:03.2f} {:03.2f} ".format(*self.r.y[:3])
            txt += "| {:03.2f} {:03.2f} {:03.2f} ".format(*self.r.y[3:])
            sys.stdout.write("\r" + txt)
            sys.stdout.flush()
            self.pts.append(gp_Pnt(*self.r.y[:3]))
            self.vel.append(gp_Vec(*self.r.y[3:]))
            self.check_ground()
            self.r.integrate(self.r.t + self.dt)

        print()
        poly = make_polygon(self.pts)
        self.display.DisplayShape(poly)
        for pnt in self.pts[::10]:
            self.display.DisplayShape(pnt)
        self.display.DisplayShape(self.grd, transparency=0.8, color="BLUE")

    def check_ground(self):
        if len(self.pts) != 1:
            ray = spl_2pnt(self.pts[-2], self.pts[-1])
            vec = gp_Vec(self.pts[-2], self.pts[-1])
            self.p_trce.Perform(ray, self.h_surf)
            if self.p_trce.NbPoints() != 0 and vec.Z() < 0:
                print()
                uvw = self.p_trce.Parameters(1)
                u, v, w = uvw
                pnt, vec = gp_Pnt(), gp_Vec()
                GeomLProp_CurveTool.D1(ray, w, pnt, vec)
                self.pts[-1] = pnt
                print(self.r.t, w, pnt)
                print(vec)
                print(gp_Vec(self.pts[-2], self.pts[-1]))
                self.r.set_initial_value(
                    self.gen_condition(pnt, gp_Vec(10, 20, 100)), self.r.t - self.dt * (1 - w))


if __name__ == '__main__':
    pnt = gp_Pnt(0, 0, 1)
    vec = gp_Vec(0, 0, 10)
    print(cnt.c, "m/s")
    print(cnt.e)
    print(cnt.g)
    print(cnt.m_e, cnt.m_n, cnt.m_p, cnt.m_u)

    obj = Particle()
    obj.compute_trajectory(t1=200)
    #obj.show_ellipsoid(rxyz=[6100 * 10**3, 6100 * 10**3, 6000 * 10**3])
    obj.show_axs_pln(obj.grd_axs, scale=25)
    obj.show_axs_pln(scale=50)
    obj.show()
