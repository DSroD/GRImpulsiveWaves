import numpy as np

from grimpulsivewaves.coordinates import DeSitterNullTetrad
from grimpulsivewaves.plotting.dynamic import PlotlyDynamicPlotter
from grimpulsivewaves.waves import LambdaGeneralSolution


def surpressor(x, sp):
    if sp is None:
        return x
    _x = x
    if hasattr(sp, "__len__"):
        for ind in sp:
            _x[ind] = 0
    else:
        _x[sp] = 0
    return _x

def to5DdS(x, lmb, surpress=None):
    eps = np.sign(lmb)
    _a = list(map(lambda y: surpressor(y, surpress), x))
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[1] + x[0]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.real(x[1] - x[0]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.real(x[2] + x[3]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.imag(x[2] - x[3]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               np.sqrt(3./(eps * lmb)) * (1 + lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))/
                               (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))], _a))


def H(x, l):
    return 0.


def H_z(x, l):
    return 0.




plotter = PlotlyDynamicPlotter(aspectratio=[1, 1, 1], labels=["Z4", "Z1", "Z0"], xrange=[-3, 3], yrange=[-3, 3],
                               zrange=[-3, 3], showSpikes=True, bgcolor="#d1f1ff")

plotter2 = PlotlyDynamicPlotter(aspectratio=[1, 1, 1], labels=["Z4", "Z1", "Z0"], xrange=[-3, 3], yrange=[-3, 3],
                               zrange=[-3, 3], showSpikes=True, bgcolor="#d1f1ff")

plotter.plotHyperboloid(1, (-1.7, 1.7), opacity=0.8, color="rgb(247,161,255)", drawImpulse=False, showlegend=True, drawCoords=True)
plotter2.plotHyperboloid(-1, (-1.7, 1.7), opacity=0.8, color="rgb(247,161,255)", drawImpulse=False, showlegend=True, drawCoords=True)


"""
N = 10

lmb = 1

initposu = [DeSitterNullTetrad(np.array([0, theta, 0j, 0j])) for theta in np.linspace(-1, 1, num=N)]
initposv = [DeSitterNullTetrad(np.array([theta, 0, 0j, 0j])) for theta in np.linspace(-1, 1, num=N)]

uu = [np.array([0.5, 0, 0, 0]) for theta in np.linspace(-2, 2, num=N)]
uv = [np.array([0, 0.5, 0, 0]) for theta in np.linspace(-2, 2, num=N)]
initvelsu = [DeSitterNullTetrad(x, dif=True) for x in uu] #Can be generalized to different initial 4-vels
initvelsv = [DeSitterNullTetrad(x, dif=True) for x in uv] #Can be generalized to different initial 4-vels
solp = LambdaGeneralSolution(lmb, H, H_z)
solm = LambdaGeneralSolution(-lmb, H, H_z)

for xu0, uu0, xv0, uv0 in zip(initposu, initvelsu, initposv, initvelsv):
    for plt, cosm in zip([plotter, plotter2], [lmb, -lmb]):

        tu = solp.generate_geodesic(xu0, uu0,(-4, 4), max_step=0.5, christoffelParams=[cosm])
        tv = solp.generate_geodesic(xv0, uv0, (-4, 4), max_step=0.5, christoffelParams=[cosm])

        tum, tup = tu[0]
        ttum, ttup = tu[1]
        tvm, tvp = tv[0]
        ttvm, ttvp = tv[1]

        plt.plotTrajectory3D(to5DdS(tum, cosm), color="rgb(20, 10, 0)", xc=4, yc=1, zc=0, name="Geod (-)", t=ttum, opacity=0.5)
        plt.plotTrajectory3D(to5DdS(tup, cosm), color="rgb(0, 10, 20)", xc=4, yc=1, zc=0, name="Geod (+)", t=ttup, opacity=0.5)

        plt.plotTrajectory3D(to5DdS(tvm, cosm), color="rgb(20, 10, 0)", xc=4, yc=1, zc=0, name="Geod (-)", t=ttvm, opacity=0.5)
        plt.plotTrajectory3D(to5DdS(tvp, cosm), color="rgb(0, 10, 20)", xc=4, yc=1, zc=0, name="Geod (+)", t=ttvp, opacity=0.5)
"""


plotter.export_html("hyp+.html", include_plotlyjs=True, include_mathjax=True)
plotter.export_pdf("hyp+.pdf", eye=(2.9, 0.5, 0.8))

plotter2.export_html("hyp-.html", include_plotlyjs=True, include_mathjax=True)
plotter2.export_pdf("hyp-.pdf", eye=(2.9, 0.7, 0.8))