from grimpulsivewaves.waves import LambdaGeneralSolution
from grimpulsivewaves.coordinates import DeSitterNullTetrad
from grimpulsivewaves.plotting import PlotlyDynamicPlotter
from grimpulsivewaves.waves import Solution

import numpy as np

import random

def r(phi):
    # return 2 - 2 * np.sin(phi) + np.sin(phi) * np.sqrt(np.abs(np.cos(phi))) / (np.sin(phi) + 1.4) # Shape of heart
    # return 4 + np.cos(phi) + np.sin(6*phi) + np.cos(2*phi + 1) # "Random shape"
    return 1 # Circle

def toConformalCart(x):
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]),
                               -1./np.sqrt(2.) * np.real(x[0] - x[1]),
                               1./np.sqrt(2.) * np.real(x[2] + x[3]),
                               1./np.sqrt(2.) * np.imag(x[2] - x[3])], x))

def toConformalUVxy(x):
    return list(map(lambda x: [np.real(x[0]),
                               np.real(x[1]),
                               1. / np.sqrt(2.) * np.real(x[2] + x[3]),
                               1. / np.sqrt(2.) * np.imag(x[2] - x[3])], x))
def toUVfrom5dS(x, lmb):
    a = np.sqrt(3./(np.sign(lmb) * lmb))
    return list(map(lambda x: [np.sqrt(2) * a * (x[0] - x[1]) / (x[4] + a),
                               np.sqrt(2) * a * (x[0] + x[1]) / (x[4] + a),
                               np.sqrt(2) * a * (x[2] + 1j * x[3]) / (x[4] + a),
                               np.sqrt(2) * a * (x[2] - 1j * x[3]) / (x[4] + a)], x))

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

def thetaU(x):
    if x[0] > 0:
        return 1.
    return 0.

def to5DdSCut(x, lmb, h0, surpress=None):
    eps = np.sign(lmb)
    _a = list(map(lambda y: surpressor(y, surpress), x))
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[1] + x[0]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))) + 1. / np.sqrt(2.) * thetaU(x) * h0,
                               1./np.sqrt(2.) * np.real(x[1] - x[0]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))) - 1. / np.sqrt(2.) * thetaU(x) * h0,
                               1./np.sqrt(2.) * np.real(x[2] + x[3]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.imag(x[3] - x[2]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               np.sqrt(3./(eps * lmb)) * (1 + lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))/
                               (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))], _a))[1:] #Ommit first so there is no line in the middle of the cut


def H(x, l):
    return 2. * np.exp(np.sqrt(- l / 3.) * (x[2] + x[3])) * (1 + 1./6. * x[2] * x[3] * l) / 2.


def H_z(x, l):
    return 2./36. * np.exp((x[2]+x[3]) * np.sqrt(- l / 3.)) * (6. * np.sqrt(-3. * l) + x[3] * l * (3. + np.sqrt(-3. * l) * x[2]))


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2 * sigma**2))

def genRGB(i, n, add = 0):
    mu = n / 2.
    sigma = mu / 2.
    return [gaussian((i + add) % n, mu, sigma) * 255., 20., (1.-gaussian((i + add) % n, mu, sigma)) * 255.]


N = 10 #Number of geodesics
mu = 1.0
lmb = -1.0

initpos = [DeSitterNullTetrad(np.array([0, theta, 0j, 0j])) for theta in np.linspace(-.2, .2, num=N)]

surps = [] #surpress


#u0 = [np.array([1, 1, 0, 0]) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
u0 = [np.array([0.1, 0, 0, 0]) for theta in np.linspace(-2, 2, num=N)]
initvels = [DeSitterNullTetrad(x, dif=True) for x in u0] #Can be generalized to different initial 4-vels

wave = LambdaGeneralSolution(lmb, H, H_z) #Generate spacetime with wave


plotter = PlotlyDynamicPlotter(title=r"$H = \exp\left(-\frac{\Lambda}{3} (\eta + \bar{\eta}) \right)(1+\frac{1}{6} \Lambda \eta \bar{\eta}),~~ \Lambda=" + str(lmb) +"$",
                               aspectratio=[1, 1, 1], labels=["Z4", "Z1", "Z0"],
                               xrange=[-3, 3], yrange=[-3, 3], zrange=[-3, 3], showSpikes=True, bgcolor="#d1f1ff") #Init 3D plotter

plotter.plotHyperboloid(lmb, (-7, 7), opacity=0.2, color="rgb(181,0,136)", drawImpulse=True, showlegend=True)


plotter2 = PlotlyDynamicPlotter(title=r"$H = \exp\left(-\frac{\Lambda}{3} (\eta + \bar{\eta}) \right)(1+\frac{1}{6} \Lambda \eta \bar{\eta}),~~ \Lambda=" + str(lmb) +"$",
                               aspectratio=[1, 1, 1], labels=["Z4", "Z1", "Z0"],
                               xrange=[-3, 3], yrange=[-3, 3], zrange=[-3, 3], showSpikes=True, bgcolor="#d1f1ff") #Init 3D plotter

plotter2.plotCutAndPasteHyperboloid(1., lmb, (-7, 7), opacity=0.2, color="rgb(181,0,136)", drawImpulse=True, showlegend=True)
# For each init pos generate geodesic (splitted)

for x0, u0, geonum in zip(initpos, initvels, range(0, len(initpos))):
    a = wave.generate_geodesic(x0, u0, (-20, 25), max_step=0.1, christoffelParams=[lmb], rtol=1e-7, atol=1e-9)
    trajm, trajp = a[0]
    tm, tp = a[1]
    #trajp = ds.generate_geodesic(x0, u0, (-20, 20), max_step=0.4, christoffelParams=[lmb])
    c = genRGB(geonum, 2*N, add=10)
    color = "rgb(" + str(c[0]) + "," + str(c[1]) + "," + str(c[2]) + ")"
    # TODO: Add name to trajectory

    plotter.plotTrajectory3D(to5DdS(trajm, lmb, surpress=surps), color=color, xc=4, yc=1, zc=0,
                             name="Geod " + str(geonum) + " (-)", t=tm)
    plotter.plotTrajectory3D(to5DdS(trajp, lmb, surpress=surps), color=color, xc=4, yc=1, zc=0,
                             name="Geod " + str(geonum) + " (+)", t=tp)

    plotter2.plotTrajectory3D(to5DdSCut(trajm, lmb, 1., surpress=surps), color=color, xc=4, yc=1, zc=0,
                             name="Geod " + str(geonum) + " (-)", t=tm)
    plotter2.plotTrajectory3D(to5DdSCut(trajp, lmb, 1., surpress=surps), color=color, xc=4, yc=1, zc=0,
                             name="Geod " + str(geonum) + " (+)", t=tp)


plotter.export_html("GenAdSExp410_NOSUP.html", include_plotlyjs=True, include_mathjax=True)
plotter.export_pdf("GenAdSExp410_NOSUP.pdf", eye=(2.5, 0.5, 0.2))

plotter2.export_html("GenAdSExp410_CutAndPaste_NOSUP.html", include_plotlyjs=True, include_mathjax=True)
plotter2.export_pdf("GenAdSExp410_CutAndPaste_NOSUP.pdf", eye=(2.5, 0.5, 0.2))
