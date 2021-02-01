from grimpulsivewaves.waves import HottaTanakaSolution
from grimpulsivewaves.coordinates import DeSitterNullTetrad

from grimpulsivewaves.plotting import PlotlyDynamicPlotter
from grimpulsivewaves.plotting import StaticGeodesicPlotter

import numpy as np

lmb = 1
mu = 1

folder = "HottaTanaka"
fname = "HT"
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

def to5DUVdS(x, lmb):
    eps = np.sign(lmb)
    return list(map(lambda x: [
        np.real(x[0]) / (1 - lmb / 6. * (np.real(x[0] * x[1]) - np.real(x[2] * x[3]))),
        np.real(x[1]) / (1 - lmb / 6. * (np.real(x[0] * x[1]) - np.real(x[2] * x[3]))),
        1. / np.sqrt(2.) * np.real(x[2] + x[3]) / (1 - lmb / 6. * (np.real(x[0] * x[1]) - np.real(x[2] * x[3]))),
        1. / np.sqrt(2.) * np.imag(x[2] - x[3]) / (1 - lmb / 6. * (np.real(x[0] * x[1]) - np.real(x[2] * x[3]))),
        np.sqrt(3. / (eps * lmb)) * (1 + lmb / 6. * (np.real(x[0] * x[1]) - np.real(x[2] * x[3]))) /
        (1 - lmb / 6. * (np.real(x[0] * x[1]) - np.real(x[2] * x[3])))], x))


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2 * sigma**2))

def quadratic(x, x0, k1, k2):
    return k1 * (x - x0)**2 + k2

def genRGB(i, n, add = 0):
    mu = n / 2.
    sigma = mu / 2.
    return (gaussian((i + add) % n, mu, sigma), 0.1, (1.-gaussian((i + add) % n, mu, sigma)))


initp = [DeSitterNullTetrad(np.array([0, 0, x+0j, x+0j])) for x in np.linspace(-1.5, 1.5, num=15) if np.abs(x) > 0.3]
initv = [DeSitterNullTetrad(np.array([1, 0, 0 + 0j, 0 + 0j]), dif=True) for x in np.linspace(-1.5, 1.5, num=15) if np.abs(x) > 0.3]

wave = HottaTanakaSolution(lmb, mu)

statplotterUV = StaticGeodesicPlotter(labels2d=[r"$\mathcal{V}$", r"$\mathcal{U}$"], aspect="auto")
statplotterxU = StaticGeodesicPlotter(labels2d=[r"$x$", r"$\mathcal{U}$"], aspect="auto")
dynplotterxyU = PlotlyDynamicPlotter(labels=["U", "x", "y"], showSpikes=False, aspectratio=[1, 1, 1])
dynplotterZ2Z3Z4 = PlotlyDynamicPlotter(labels=["Z2", "Z3", "Z4"], showSpikes=False, aspectratio=[1, 1, 1])
dynplotterZ0Z2Z4 = PlotlyDynamicPlotter(labels=["Z0", "Z2", "Z4"], showSpikes=False, aspectratio=[1, 1, 1])

for x0, u0, geonum in zip(initp, initv, range(0, len(initp))):
    fr = -quadratic(geonum, len(initp)/2, 0.1/len(initp), 0.9)
    to = quadratic(geonum, len(initp)/2, 0.2/len(initp), 0.5)
    a = wave.generate_geodesic(x0, u0, (fr, to), max_step=0.005, christoffelParams=[lmb])
    print("Integrating geo {} form {} to {}".format(geonum, fr, to))
    print("Initial conditions are: x = {}   ;u = {}".format(x0.x, u0.x))
    trajm, trajp = a[0]
    tm, tp = a[1]
    color = genRGB(geonum, len(initp))
    #UV plot
    statplotterUV.plot(toConformalUVxy(trajm), xc=1, yc=0)
    statplotterUV.plot(toConformalUVxy(trajp), xc=1, yc=0, line="-", color=color)

    #Ux plot
    statplotterxU.plot(toConformalUVxy(trajm), xc=2, yc=0, color=color)
    statplotterxU.plot(toConformalUVxy(trajp), xc=2, yc=0, line="-", color=color)

    #Uxy plot
    dynplotterxyU.plotTrajectory3D(toConformalUVxy(trajm), xc=0, yc=2, zc=3, t=tm, dash="longdashdot", linewidth=4)
    dynplotterxyU.plotTrajectory3D(toConformalUVxy(trajp), xc=0, yc=2, zc=3, t=tp, linewidth=4)

    #Z0Z2Z4
    dynplotterZ0Z2Z4.plotTrajectory3D(to5DUVdS(trajm, lmb), xc=0, yc=2, zc=4, t=tm, dash="longdashdot", linewidth=4)
    dynplotterZ0Z2Z4.plotTrajectory3D(to5DUVdS(trajp, lmb), xc=0, yc=2, zc=4, t=tp, linewidth=4)

    #Z2Z3Z4
    dynplotterZ2Z3Z4.plotTrajectory3D(to5DUVdS(trajm, lmb), xc=2, yc=3, zc=4, t=tm, dash="longdashdot", linewidth=4)
    dynplotterZ2Z3Z4.plotTrajectory3D(to5DUVdS(trajp, lmb), xc=2, yc=3, zc=4, t=tp, linewidth=4)

statplotterUV.save("{}/{}_UV_mu{}_lmb{}".format(folder, fname, mu, lmb), dpi=500)
statplotterxU.save("{}/{}_xU_mu{}_lmb{}".format(folder, fname, mu, lmb), dpi=500)

dynplotterxyU.export_html("{}/{}_x-y-U_mu{}_lmb{}.html".format(folder, fname, mu, lmb))
dynplotterxyU.export_pdf("{}/{}_x-y-U_mu{}_lmb{}.pdf".format(folder, fname, mu, lmb), title=False)

dynplotterZ0Z2Z4.export_html("{}/{}_Z0-Z2-Z4_mu{}_lmb{}.html".format(folder, fname, mu, lmb))
dynplotterZ0Z2Z4.export_pdf("{}/{}_Z0-Z2-Z4_mu{}_lmb{}.pdf".format(folder, fname, mu, lmb), title=False)

dynplotterZ2Z3Z4.export_html("{}/{}_Z2-Z3-Z4_mu{}_lmb{}.html".format(folder, fname, mu, lmb))
dynplotterZ2Z3Z4.export_pdf("{}/{}_Z2-Z3-Z4_mu{}_lmb{}.pdf".format(folder, fname, mu, lmb), title=False)
