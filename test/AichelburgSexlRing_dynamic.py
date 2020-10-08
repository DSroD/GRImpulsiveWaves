import numpy as np
from grimpulsivewaves.coordinates.coordinates import NullTetradAichelburgSexlGyraton
from grimpulsivewaves.waves import AichelburgSexlGyratonSolution
from grimpulsivewaves.plotting import PlotlyDynamicPlotter

import random

def r(phi):
    return 0.15 * np.sin(6 * phi) + 0.2 * np.cos(11 * phi)  + 0.1 * np.sin(phi/2) + 1

N = 80 #Number of geodesics
r2 = 2
mu = 1
chi = 2 * np.pi

initpos = [NullTetradAichelburgSexlGyraton(np.array([0, 0, r(phi) * np.exp(1j * phi), r(phi) * np.exp(-1j * phi)]))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]

#initpos = [NullTetradAichelburgSexlGyraton(np.array([]))]
u0 = np.array([1, 1, 0, 0])
initvels = [NullTetradAichelburgSexlGyraton(u0 / np.linalg.norm(u0), True) for u in range(N)] #Can be generalized to different initial 4-vels

wave = AichelburgSexlGyratonSolution(mu, chi) #Generate spacetime with wave

plotter = PlotlyDynamicPlotter(title="Gyratonic Aichelburg Sexl solution, mu=" + str(mu) +", chi=2*pi", aspectratio=[1, 1, 1], labels=["x", "y", "t"], xrange=[-30,30], yrange=[-30,30]) #Init 3D plotter

# plotter.plotHyperboloid(2)

def toGyraCart(x):
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]),
                               1./np.sqrt(2.) * np.real(x[0] - x[1]),
                               1./np.sqrt(2.) * np.real(x[2] + x[3]),
                               1./np.sqrt(2.) * np.imag(x[2] - x[3])], x))

def toUVxy(x):
    return list(map(lambda x: [x[0],
                               x[1],
                               1. / np.sqrt(2.) * np.real(x[2] + x[3]),
                               1. / np.sqrt(2.) * np.imag(x[2] - x[3])], x))

# For each init pos generate geodesic (splitted)

for x0, u0, geonum in zip(initpos, initvels, range(0, len(initpos))):
    trajm, trajp = wave.generate_geodesic(x0, u0, (-0.3, 0.5), max_step=0.005, christoffelParams=[chi])
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()
    # TODO: Add name to trajectory

    plotter.plotTrajectory3D(toGyraCart(trajm), color=color, xc=2, yc=3, zc=0)

    plotter.plotTrajectory3D(toGyraCart(trajp), color=color, xc=2, yc=3, zc=0)


plotter.show()

plotter.export_html("gyratonic2.html", include_plotlyjs=False)

