import numpy as np
from grimpulsivewaves.coordinates.coordinates import NullTetradConstantHeavisideGyraton
from grimpulsivewaves.waves import AichelburgSexlGyratonSolution
from grimpulsivewaves.plotting import PlotlyDynamicPlotter


import random

def r(phi):
    # return 2 - 2 * np.sin(phi) + np.sin(phi) * np.sqrt(np.abs(np.cos(phi))) / (np.sin(phi) + 1.4) # Shape of heart
    # return 4 + np.cos(phi) + np.sin(6*phi) + np.cos(2*phi + 1) # "Random shape"
     return 1 # Circle

N = 40 #Number of geodesics
r2 = 2
mu = 1
ch = -4
chi = ch * np.pi

initpos = [NullTetradConstantHeavisideGyraton(np.array([0, 0, r(phi) * np.exp(1j * phi), r(phi) * np.exp(-1j * phi)]))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]


u0 = [np.array([-1, 1, 0.2 * np.exp(1j * phi), 0.2 * np.exp(-1j * phi)]) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
initvels = [NullTetradConstantHeavisideGyraton(x, dif=True) for x in u0] #Can be generalized to different initial 4-vels

wave = AichelburgSexlGyratonSolution(mu, chi) #Generate spacetime with wave

def toGyraCart(x):
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]),
                               -1./np.sqrt(2.) * np.real(x[0] - x[1]),
                               1./np.sqrt(2.) * np.real(x[2] + x[3]),
                               1./np.sqrt(2.) * np.imag(x[2] - x[3])], x))

def toGyraUVxy(x):
    return list(map(lambda x: [np.real(x[0]),
                               np.real(x[1]),
                               1. / np.sqrt(2.) * np.real(x[2] + x[3]),
                               1. / np.sqrt(2.) * np.imag(x[2] - x[3])], x))


plotter = PlotlyDynamicPlotter(title=r"$\text{Gyratonic Aichelburg Sexl solution, }\mu=" + str(mu) +", \chi=" + str(ch) +"\pi$",
                               aspectratio=[1, 1, 1], labels=["x", "y", "z"],
                               xrange=[-10, 10], yrange=[-10, 10], zrange=[-5, 20]) #Init 3D plotter


# plotter.plotHyperboloid(6)

# For each init pos generate geodesic (splitted)

for x0, u0, geonum in zip(initpos, initvels, range(0, len(initpos))):
    a = wave.generate_geodesic(x0, u0, (-3, 5), max_step=0.3, christoffelParams=[chi])
    trajm, trajp = a[0]
    tm, tp = a[1]
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()
    # TODO: Add name to trajectory

    plotter.plotTrajectory3D(toGyraCart(trajm[0]), color=color, xc=2, yc=3, zc=1, name="Geodesic (-)")

    plotter.plotTrajectory3D(toGyraCart(trajp[0]), color=color, xc=2, yc=3, zc=1, name="Geodesic (+)")

def _highlight_trace(trace, points, selector):
    if len(points.point_inds) == 0:
        return

    for i, _ in enumerate(plotter.fig.data):
        plotter.fig.data[i]['line']['width'] = 2 + 20 * (i == points.trace_index)
        if i%2==0:
            j = i+1
        else:
            j= i-1
        plotter.fig.data[j]['line']['width'] = 2 + 20 * (i == points.trace_index)


for i in range(0, len(plotter.fig.data)):
    plotter.fig.data[i].on_hover(_highlight_trace)

plotter.show()

plotter.export_html("002.html", include_plotlyjs=True, include_mathjax=True)

