import numpy as np
from grimpulsivewaves.coordinates import NullTetradConstantHeavisideGyraton
from grimpulsivewaves.coordinates import NullTetrad
from grimpulsivewaves.waves import AichelburgSexlGyratonSolution
from grimpulsivewaves.waves import AichelburgSexlSolution
from grimpulsivewaves.plotting import PlotlyDynamicPlotter


import random

def r(phi):
    # return 2 - 2 * np.sin(phi) + np.sin(phi) * np.sqrt(np.abs(np.cos(phi))) / (np.sin(phi) + 1.4) # Shape of heart
    # return 4 + np.cos(phi) + np.sin(6*phi) + np.cos(2*phi + 1) # "Random shape"
     return 1 # Circle

N = 30 #Number of geodesics
r2 = 2
mu = -1.0
ch = 0.2
chi = ch * np.pi

initpos = [NullTetradConstantHeavisideGyraton(np.array([0, 0, r(phi) * np.exp(1j * phi), r(phi) * np.exp(-1j * phi)], dtype=np.complex128))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
initpos0 = [NullTetrad(np.array([0, 0, r(phi) * np.exp(1j * phi), r(phi) * np.exp(-1j * phi)], dtype=np.complex128))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]

def metric(u, v):
    return -u[0]*v[1] - u[1]*v[0] + u[2]*v[3] + u[3]*v[2]

u0 = [np.array([1.0, 0.0, 0j, 0j], dtype=np.complex128) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
print(metric(u0[0], u0[0]))
initvels = [NullTetradConstantHeavisideGyraton(x, dif=True) for x in u0] #Can be generalized to different initial 4-vels
initvels0 = [NullTetrad(x, dif=True) for x in u0]
print(metric(initvels[0].x, initvels[0].x))


wave = AichelburgSexlSolution(mu)
wave2 = AichelburgSexlGyratonSolution(mu, chi) #Generate spacetime with wave
wave3 = AichelburgSexlGyratonSolution(mu, 2*chi)

def toGyraCart(x):
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[1] + x[0]),
                               1./np.sqrt(2.) * np.real(x[1] - x[0]),
                               1./np.sqrt(2.) * np.real(x[2] + x[3]),
                               1./np.sqrt(2.) * np.imag(x[2] - x[3])], x))

def toGyraUVxy(x):
    return list(map(lambda x: [np.real(x[0]),
                               np.real(x[1]),
                               1. / np.sqrt(2.) * np.real(x[2] + x[3]),
                               1. / np.sqrt(2.) * np.imag(x[2] - x[3])], x))


# For each init pos generate geodesic (splitted)
def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2 * sigma**2))

def genRGB(i, n, add = 0):
    mu = n / 2.
    sigma = mu / 2.
    return [gaussian((i + add) % n, mu, sigma) * 255., 0., (1.-gaussian((i + add) % n, mu, sigma)) * 255.]

plot = [2, 3, 0]
lab = ["x", "y", "t"]
convertFunction = toGyraCart
plotName = "ASRingGyraNull"

plotter = PlotlyDynamicPlotter(title=r"$\text{Aichelburg Sexl solution, }\mu=" + str(mu) + "$",
                               aspectratio=[1, 1, 1], labels=lab,
                               xrange=[-10, 10], yrange=[-10, 10], zrange=[-4, 15], bgcolor="#ccffeb")

plotterg = PlotlyDynamicPlotter(title=r"$\text{Gyratonic Aichelburg Sexl solution, }\mu=" + str(mu) +", \chi=" + str(ch) +" \pi$",
                               aspectratio=[1, 1, 1], labels=lab,
                               xrange=[-10, 10], yrange=[-10, 10], zrange=[-4, 13], bgcolor="#ccffeb")

plotterg2 = PlotlyDynamicPlotter(title=r"$\text{Gyratonic Aichelburg Sexl solution, }\mu=" + str(mu) +", \chi=" + str(2 * ch) +" \pi$",
                               aspectratio=[1, 1, 1], labels=lab,
                               xrange=[-10, 10], yrange=[-10, 10], zrange=[-4, 10], bgcolor="#ccffeb") #Init 3D plotter



for x0g, u0g, x0, u0, geonum in zip(initpos, initvels, initpos0, initvels0, range(0, len(initpos))):
    print("")
    print("GEO {}, 0".format(geonum))
    a = wave.generate_geodesic(x0, u0, (-4, 5), max_step=0.2)
    print("")
    print("GEO {}, 1".format(geonum))
    ag = wave2.generate_geodesic(x0g, u0g, (-4, 5), max_step=0.05, christoffelParams=[chi, False], christoffelParamsPlus=[chi, True])
    print("")
    print("GEO {}, 2".format(geonum))
    ag2 = wave3.generate_geodesic(x0g, u0g, (-4, 5), max_step=0.05, christoffelParams=[chi*2, False], christoffelParamsPlus=[chi*2, False])
    print("")
    print("")
    print("----------------------------------------------")

    trajm, trajp = a[0]
    tm, tp = a[1]

    trajmg, trajpg = ag[0]
    tmg, tpg = ag[1]

    trajmg2, trajpg2 = ag2[0]
    tmg2, tpg2 = ag2[1]

    rgb = genRGB(geonum, len(initpos), 15)
    color = 'rgb(' + str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2]) + ')'
    # TODO: Add name to trajectory

    plotter.plotTrajectory3D(convertFunction(trajm), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (-)", t=tm)
    plotter.plotTrajectory3D(convertFunction(trajp), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (+)", t=tp)

    plotterg.plotTrajectory3D(convertFunction(trajmg), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (-)", t=tmg)
    plotterg.plotTrajectory3D(convertFunction(trajpg), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (+)", t=tpg)

    plotterg2.plotTrajectory3D(convertFunction(trajmg2), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (-)", t=tmg2)
    plotterg2.plotTrajectory3D(convertFunction(trajpg2), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (+)", t=tpg2)


plotter.export_html(plotName + "000_" + lab[0] + lab[1] + lab[2] + ".html", include_plotlyjs=True, include_mathjax=True)
plotter.export_pdf(plotName + "000_" + lab[0] + lab[1] + lab[2] + ".pdf", eye=(2.5, 0.5, 1.2), up=(0., 0., 1.))

plotterg.export_html(plotName + "001_" + lab[0] + lab[1] + lab[2] + ".html", include_plotlyjs=True, include_mathjax=True)
plotterg.export_pdf(plotName + "001_" + lab[0] + lab[1] + lab[2] + ".pdf", eye=(2.5, 0.5, 1.2), up=(0., 0., 1.))

plotterg2.export_html(plotName + "002_" + lab[0] + lab[1] + lab[2] + ".html", include_plotlyjs=True, include_mathjax=True)
plotterg2.export_pdf(plotName + "002_" + lab[0] + lab[1] + lab[2] + ".pdf", eye=(2.5, 0.5, 1.2), up=(0., 0., 1.))
