import numpy as np

from grimpulsivewaves.coordinates import NullTetrad

from grimpulsivewaves.waves import AichelburgSexlSolution

from grimpulsivewaves.plotting import PlotlyDynamicPlotter
from grimpulsivewaves.plotting import StaticGeodesicPlotter


import random

def r(phi):
    # return 2 - 2 * np.sin(phi) + np.sin(phi) * np.sqrt(np.abs(np.cos(phi))) / (np.sin(phi) + 1.4) # Shape of heart
    # return 4 + np.cos(phi) + np.sin(6*phi) + np.cos(2*phi + 1) # "Random shape"
    return 0.5 + phi / (2 * np.pi) # Circle

N = 25 #Number of geodesics
r2 = 2
mu = np.array([0.5, 2, 4])
col = ["g", "r", "teal"]

u0 = [np.array([0.5, 1, 0, 0], dtype=np.complex128) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
initvels = [NullTetrad(x, dif=True) for x in u0]


initpos = [NullTetrad(np.array([0, 0, r(phi), r(phi)], dtype=np.complex128))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]


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
lab = ["x", "y", "U"]
convertFunction = toGyraUVxy
plotName = "AS/ASMatterLine"


staticplotter = StaticGeodesicPlotter(labels2d = [r"$\mathcal{U}$", r"$\mathcal{V}$"], labelsize=16, aspect='auto', figsize=(6,2.1))

for m, i in zip(mu, range(len(mu))):
    wave = AichelburgSexlSolution(m)
    plotter = PlotlyDynamicPlotter(title="",
                                   aspectratio=[1, 1, 1], labels=lab,
                                   xrange=[-6, 6], yrange=[-6, 6], zrange=[-2, 4],
                                   bgcolor="#ccffeb", fontsize=30)

    staticplotter2 = StaticGeodesicPlotter(labels2d=["x", "z"], labelsize=16, aspect='auto',
                                          figsize=(16, 12))

    for x0, u0, geonum in zip(initpos, initvels, range(0, len(initpos))):

        a = wave.generate_geodesic(x0, u0, (-3, 3), max_step=0.2, verbose=False)

        trajm, trajp = a[0]
        tm, tp = a[1]

        rgb = genRGB(geonum, len(initpos), 8)
        rgb2 = genRGB(geonum, 2*len(initpos), len(initpos))
        color = 'rgb(' + str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2]) + ')'
        # TODO: Add name to trajectory

        plotter.plotTrajectory3D(convertFunction(trajm), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic {} (-)".format(geonum), t=tm, linewidth=4, dash="longdashdot")
        plotter.plotTrajectory3D(convertFunction(trajp), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geodesic {} (+)".format(geonum), t=tp, linewidth=4)

        #staticplotter2.plot(toGyraCart(trajm), "-", color="k", xc=2, yc=1)
        staticplotter2.plot(toGyraCart(trajp), "-", color=np.array(rgb2)/255., xc=2, yc=1)

        if(geonum == 0):
            staticplotter.plot(toGyraUVxy(trajm), "-", color="k", xc=0, yc=1)
            staticplotter.plot(toGyraUVxy(trajp), "-", color=col[i], xc=0, yc=1, label="$b_0$ = {}".format(m))


    plotter.export_html("{}_{}{}{}_mu={}.html".format(plotName, lab[0], lab[1], lab[2], m), include_plotlyjs=True, include_mathjax=True)
    plotter.export_pdf("{}_{}{}{}_mu={}.pdf".format(plotName, lab[0], lab[1], lab[2], m), eye=(1.4, 1.4, 1.3), up=(0., 0., 1.))
    staticplotter2.ax.plot([0, 0], [-5, 5], "k-")
    staticplotter2.ax.set_ylim([-0.6, 2.5])
    staticplotter2.save("{}_xz_mu={}.png".format(plotName,m), dpi=700, showlegend=False)

staticplotter.save("{}_UV.png".format(plotName), dpi=500, showlegend=True)

