import numpy as np
from grimpulsivewaves.coordinates import NullTetradConstantHeavisideGyraton
from grimpulsivewaves.coordinates import NullTetrad
from grimpulsivewaves.waves import GeneralGyratonicRefractionSolution
from grimpulsivewaves.waves import GeneralRefractionSolution
from grimpulsivewaves.plotting import PlotlyDynamicPlotter


import random

def r(phi):
    # return 2 - 2 * np.sin(phi) + np.sin(phi) * np.sqrt(np.abs(np.cos(phi))) / (np.sin(phi) + 1.4) # Shape of heart
    # return 4 + np.cos(phi) + np.sin(6*phi) + np.cos(2*phi + 1) # "Random shape"
     return 1 # Circle

def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2 * sigma**2))

def genRGB(i, n, add = 0):
    mu = n / 2.
    sigma = mu / 2.
    return [gaussian((i + add) % n, mu, sigma) * 255., 0., (1.-gaussian((i + add) % n, mu, sigma)) * 255.]

def toGyraCart(xi, ti):
    return list(map(lambda x, t: [1./np.sqrt(2.) * np.real(x[1] + x[0]),
                                  1./np.sqrt(2.) * np.real(x[1] - x[0]),
                                  1./np.sqrt(2.) * np.real(x[2] + x[3]),
                                  1./np.sqrt(2.) * np.imag(x[2] - x[3]),
                                  t], xi, ti))

def toGyraUVxy(xi, ti):
    return list(map(lambda x, t: [np.real(x[0]),
                                  np.real(x[1]),
                                  1. / np.sqrt(2.) * np.real(x[2] + x[3]),
                                  1. / np.sqrt(2.) * np.imag(x[2] - x[3]),
                                  t], xi, ti))

N = 30 #Number of geodesics
ch = np.array([0.1, 0.2, 0.3, 0.4])
chi = ch

initpos = [NullTetradConstantHeavisideGyraton(np.array([0, 0, 1 * np.exp(-1j * phi), 1 * np.exp(1j * phi)], dtype=np.complex128))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
initpos0 = [NullTetrad(np.array([0, 0, 1 * np.exp(-1j * phi), 1 * np.exp(1j * phi)], dtype=np.complex128))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]


u0 = [np.array([1.0, 1.0, 0j, 0j], dtype=np.complex128) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
initvels = [NullTetradConstantHeavisideGyraton(x, dif=True) for x in u0] #Can be generalized to different initial 4-vels
initvels0 = [NullTetrad(x, dif=True) for x in u0]

def H(x, arg): #x^2 - y^2
    return -arg[0] * (x[2]*x[2] + x[3] * x[3])

def H_z(x, arg):
    return -2 * arg[0] * (x[2])

plot = [2, 3, 1]
lab = ["x", "y", "V"]
convertFunction = toGyraUVxy


plot2 = [2, 3, 0]
lab2 = ["x", "y", "U"]
convertFunction2 = toGyraUVxy

plotName = "xsqr_ysqr/x_xsqr_ysqr_solution"

plotter = PlotlyDynamicPlotter(title="",
                               aspectratio=[1, 1, 1], labels=lab,
                               xrange=[-10, 10], yrange=[-10, 10], zrange=[-4, 15], bgcolor="#ccffeb")

plotter2 = PlotlyDynamicPlotter(title="",
                               aspectratio=[1, 1, 1], labels=lab2,
                               xrange=[-10, 10], yrange=[-10, 10], zrange=[-4, 15], bgcolor="#ccffeb")

wave1 = GeneralRefractionSolution(H, H_z, .2)
wave2 = GeneralGyratonicRefractionSolution(H, H_z, chi[0], .2) #Generate spacetime with wave

plotterg = []
plotterg2 = []
for chival in chi:
    plotterg.append(PlotlyDynamicPlotter(title="", aspectratio=[1, 1, 1], labels=lab, xrange=[-10, 10], yrange=[-10, 10], zrange=[-4, 13], bgcolor="#ccffeb"))

    plotterg2.append(PlotlyDynamicPlotter(title="", aspectratio=[1, 1, 1], labels=lab2, xrange=[-10, 10], yrange=[-10, 10], zrange=[-4, 13], bgcolor="#ccffeb"))


for x0g, u0g, x0, u0, geonum in zip(initpos, initvels, initpos0, initvels0, range(0, len(initpos))):
    a = wave1.generate_geodesic(x0, u0, (-4, 5), max_step=0.2)


    trajm, trajp = a[0]
    tm, tp = a[1]


    rgb = genRGB(geonum, len(initpos), 15)
    color = 'rgb(' + str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2]) + ')'
    # TODO: Add name to trajectory

    plotter.plotTrajectory3D(convertFunction(trajm, tm), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (-)", t=tm)
    plotter.plotTrajectory3D(convertFunction(trajp, tp), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (+)", t=tp)

    plotter2.plotTrajectory3D(convertFunction2(trajm, tm), color=color, xc=plot2[0], yc=plot2[1], zc=plot2[2],
                             name="Geodesic (-)", t=tm)
    plotter2.plotTrajectory3D(convertFunction2(trajp, tp), color=color, xc=plot2[0], yc=plot2[1], zc=plot2[2],
                             name="Geodesic (+)", t=tp)


    for chival, p1, p2 in zip(chi, plotterg, plotterg2):
        wave2.chi = chival #this is ugly but whatever, right?
        ag = wave2.generate_geodesic(x0g, u0g, (-4, 5), max_step=0.05, christoffelParams=[chival, False],
                                     christoffelParamsPlus=[chival, True])
        trajmg, trajpg = ag[0]
        tmg, tpg = ag[1]

        p1.plotTrajectory3D(convertFunction(trajmg, tmg), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (-)", t=tmg)
        p1.plotTrajectory3D(convertFunction(trajpg, tpg), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic (+)", t=tpg)

        p2.plotTrajectory3D(convertFunction2(trajmg, tmg), color=color, xc=plot2[0], yc=plot2[1], zc=plot2[2], name="Geodesic (-)", t=tmg)
        p2.plotTrajectory3D(convertFunction2(trajpg, tpg), color=color, xc=plot2[0], yc=plot2[1], zc=plot2[2], name="Geodesic (+)", t=tpg)


plotter.export_html(plotName + "000_" + lab[0] + lab[1] + lab[2] + ".html", include_plotlyjs=True, include_mathjax=True)
plotter.export_pdf(plotName + "000_" + lab[0] + lab[1] + lab[2] + ".pdf", eye=(2.5, 0.5, 1.2), up=(0., 0., 1.))

plotter2.export_html(plotName + "000_" + lab2[0] + lab2[1] + lab2[2] + ".html", include_plotlyjs=True, include_mathjax=True)
plotter2.export_pdf(plotName + "000_" + lab2[0] + lab2[1] + lab2[2] + ".pdf", eye=(2.5, 0.5, 1.2), up=(0., 0., 1.))

for chival, g, g2 in zip(chi, plotterg, plotterg2):
    g.export_html(plotName + "001_" + lab[0] + lab[1] + lab[2] + "_chi" + str(chival) + ".html", include_plotlyjs=True, include_mathjax=True)
    g.export_pdf(plotName + "001_" + lab[0] + lab[1] + lab[2] + "_chi" + str(chival) + ".pdf", eye=(2.5, 0.5, 1.2), up=(0., 0., 1.))

    g2.export_html(plotName + "001_" + lab2[0] + lab2[1] + lab2[2] + "_chi" + str(chival) + ".html", include_plotlyjs=True, include_mathjax=True)
    g2.export_pdf(plotName + "001_" + lab2[0] + lab2[1] + lab2[2] + "_chi" + str(chival) + ".pdf", eye=(2.5, 0.5, 1.2), up=(0., 0., 1.))

