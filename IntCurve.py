import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cnt
import sys

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Pln
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool, GeomLProp_CurveTool
from OCC.Core.GeomAPI import GeomAPI_IntCS, GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCCUtils.Construct import make_plane

from particle.base import spl_2pnt


if __name__ == '__main__':
    p0 = gp_Pnt(10, 10, 50)
    p1 = gp_Pnt(-20, -20, -50)
    crv = spl_2pnt(p0, p1)
    pln = make_plane(vec_normal=gp_Vec(0, 1, 1))
    h_surf = BRep_Tool.Surface(pln)
    api = GeomAPI_IntCS(crv, h_surf)
    print(api.IsDone())
    print(api.NbSegments())

    uvw = api.Parameters(1)
    u, v, w = uvw
    pnt_crv = gp_Pnt()
    pnt_1st = gp_Pnt()
    pnt_2nd = gp_Pnt()
    pnt_srf = gp_Pnt()
    GeomLProp_CurveTool.Value(crv, w, pnt_crv)
    GeomLProp_CurveTool.Value(crv, 0, pnt_1st)
    GeomLProp_CurveTool.Value(crv, 1, pnt_2nd)
    GeomLProp_SurfaceTool.Value(h_surf, u, v, pnt_srf)
    print(uvw)
    print(pnt_crv)
    print(pnt_1st)
    print(pnt_2nd)
    print(pnt_srf)

    pnt_crv = gp_Pnt()
    vec_crv = gp_Vec()
    GeomLProp_CurveTool.D1(crv, w, pnt_crv, vec_crv)
    print(pnt_crv)
    print(vec_crv)
    print(gp_Vec(p0, p1))
