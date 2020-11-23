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

def to5DdS(x, lmb):
    eps = np.sign(lmb)
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.real(x[0] - x[1]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.real(x[2] + x[3]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.imag(x[2] - x[3]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               np.sqrt(3./(eps * lmb)) * (1 + lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))/
                               (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))], x))


def H(x, l):
    return 1. * np.exp(np.sqrt(- l / 3.) * (x[2] + x[3])) * (1 + 1./6. * x[2] * x[3] * l) / 2.

def H_z(x, l):
    return 1./36. * np.exp((x[2]+x[3]) * np.sqrt(- l / 3.)) * (6. * np.sqrt(-3. * l) + x[3] * l * (3 + np.sqrt(-3. * l) * x[2]))

NU = 40 #Number of geodesics
NV = 1 #ve směru V
mu = 1.0
lmb = -1.0

#initarray = [np.sin(phi), np.cos(phi), np.]

#initpos = [DeSitterNullTetrad(np.array([0, 0, r(phi) * np.exp(1j * phi), r(phi) * np.exp(-1j * phi)]))
#           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]

#gp, gt = np.meshgrid(np.linspace(-10, 10, num=N), np.linspace(-10, 10, num=N))

initpos = [DeSitterNullTetrad(np.array([np.tan(phi), np.tan(theta), 0j, 0j])) for phi in np.linspace(-1.2, 1.2, num=NU) for theta in np.linspace(0, 1.2, num=NV)]


#u0 = [np.array([1, 1, 0, 0]) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
u0 = [np.array([0, 1, 0, 0]) for phi in np.linspace(-2, 2, num=NU) for theta in np.linspace(-2, 2, num=NV)]
initvels = [DeSitterNullTetrad(x, dif=True) for x in u0] #Can be generalized to different initial 4-vels

wave = LambdaGeneralSolution(lmb, H, H_z) #Generate spacetime with wave
ds = Solution()


plotter = PlotlyDynamicPlotter(title=r"$H = \frac{1}{2}\exp\left(-\frac{\Lambda}{3} (\eta + \bar{\eta}) \right)(1+\frac{1}{6} \Lambda \eta \bar{\eta}),~~ \Lambda=" + str(lmb) +"$",
                               aspectratio=[1, 1, 1], labels=["Z4", "Z1", "Z0"],
                               xrange=[-6, 6], yrange=[-6, 6], zrange=[-6, 6], showSpikes=True) #Init 3D plotter

plotter.plotHyperboloid(lmb, (-13, 13), opacity=0.3, color="rgb(153,153,255)", drawImpulse=True, showlegend=True)

# For each init pos generate geodesic (splitted)

for x0, u0, geonum in zip(initpos, initvels, range(0, len(initpos))):
    a = wave.generate_geodesic(x0, u0, (-8, 8), max_step=0.2, christoffelParams=[lmb], rtol=1e-2, atol=1e-4)
    trajm, trajp = a[0]
    tm, tp = a[1]
    #trajp = ds.generate_geodesic(x0, u0, (-20, 20), max_step=0.4, christoffelParams=[lmb])
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()
    # TODO: Add name to trajectory

    plotter.plotTrajectory3D(to5DdS(trajm, lmb), color=color, xc=4, yc=1, zc=0, name="Geod " + str(geonum) + " (-)", t=tm)

    plotter.plotTrajectory3D(to5DdS(trajp, lmb), color=color, xc=4, yc=1, zc=0, name="Geod " + str(geonum) + " (+)", t=tp)

plotter.show()
plotter.export_html("GenAdSExp410.html", include_plotlyjs=True, include_mathjax=True)
plotter.export_pdf("GenAdSExp410.pdf")
