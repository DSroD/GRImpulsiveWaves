from grimpulsivewaves.waves import HottaTanakaSolution
from grimpulsivewaves.coordinates import DeSitterNullTetrad

from grimpulsivewaves.plotting import PlotlyDynamicPlotter
from grimpulsivewaves.plotting import StaticGeodesicPlotter

import numpy as np

lmb = -1.5
mu = 1

folder = "HottaTanaka"
fname = "HT4_y-1_"

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


def omega(x, l):
    return 1 - 1./6. * l * np.real(x[2] * x[3] - x[0] * x[1])

def g(u, v, x, l):
    return (u[2] * v[3] + u[3] * v[2] - u[0] * v[1] - u[1] * v[0]) / omega(x, l)**2


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2 * sigma**2))

def quadratic(x, x0, k1, k2):
    return k1 * (x - x0)**2 + k2

def genRGB(i, n, add = 0):
    mu = n / 2.
    sigma = mu / 2.
    return (gaussian((i + add) % n, mu, sigma), 0.1, (1.-gaussian((i + add) % n, mu, sigma)))

N = 12

initu = np.array([1, 0, 0, 0])

initp = [DeSitterNullTetrad(np.array([0, 0, x+1j, x-1j])) for x in np.linspace(-1.2, 1.2, num=N) if np.abs(x) > 0.3]
initv = [DeSitterNullTetrad(initu, dif=True) for x in np.linspace(-1.2, 1.2, num=N) if np.abs(x) > 0.3]

wave = HottaTanakaSolution(lmb, mu)

statplotterUV = StaticGeodesicPlotter(labels2d=[r"$\mathcal{V}$", r"$\mathcal{U}$"], aspect="auto", ticks=True, tick_labelsize=14)
statplotterxU = StaticGeodesicPlotter(labels2d=[r"$x$", r"$\mathcal{U}$"], aspect="auto", ticks=True, tick_labelsize=14)
dynplotterxyU = PlotlyDynamicPlotter(labels=["U", "x", "y"], showSpikes=False, aspectratio=[1, 1, 0.5], bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)
dynplotterZ2Z3Z4 = PlotlyDynamicPlotter(labels=["Z2", "Z3", "Z4"], showSpikes=False, aspectratio=[1, 1, 1], bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)
dynplotterZ0Z2Z4 = PlotlyDynamicPlotter(labels=["Z0", "Z2", "Z4"], showSpikes=False, aspectratio=[1, 1, 1], bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)

for x0, u0, geonum in zip(initp, initv, range(0, len(initp))):
    fr = -quadratic(geonum, (len(initp)-1)/2., 0.5/len(initp), 0.6)
    to = quadratic(geonum, (len(initp)-1)/2., 0.7/len(initp), 1)
    print("Integrating geo {} form {} to {}".format(geonum, fr, to))
    print("Initial conditions are: x = {}   ;u = {}".format(x0.x, u0.x))
    a = wave.generate_geodesic(x0, u0, (fr, to), max_step=0.005, christoffelParams=[lmb], verbose=True)
    trajm, trajp = a[0]
    tm, tp = a[1]
    color = genRGB(geonum+0.5, len(initp))
    #UV plot
    statplotterUV.plot(toConformalUVxy(trajm), xc=1, yc=0)
    statplotterUV.plot(toConformalUVxy(trajp), xc=1, yc=0, line="-", color=color)

    #Ux plot
    statplotterxU.plot(toConformalUVxy(trajm), xc=2, yc=0, color=color)
    statplotterxU.plot(toConformalUVxy(trajp), xc=2, yc=0, line="-", color=color)

    #Uxy plot
    dynplotterxyU.plotTrajectory3D(toConformalUVxy(trajm), xc=0, yc=2, zc=3, t=tm, dash="longdashdot", linewidth=4, name=geonum, color='rgb' + str(color))
    dynplotterxyU.plotTrajectory3D(toConformalUVxy(trajp), xc=0, yc=2, zc=3, t=tp, linewidth=4, name=geonum, color='rgb' + str(color))

    #Z0Z2Z4
    dynplotterZ0Z2Z4.plotTrajectory3D(to5DUVdS(trajm, lmb), xc=0, yc=2, zc=4, t=tm, dash="longdashdot", linewidth=4, name=geonum, color='rgb' + str(color))
    dynplotterZ0Z2Z4.plotTrajectory3D(to5DUVdS(trajp, lmb), xc=0, yc=2, zc=4, t=tp, linewidth=4, name=geonum, color='rgb' + str(color))

    #Z2Z3Z4
    dynplotterZ2Z3Z4.plotTrajectory3D(to5DUVdS(trajm, lmb), xc=2, yc=3, zc=4, t=tm, dash="longdashdot", linewidth=4, name=geonum, color='rgb' + str(color))
    dynplotterZ2Z3Z4.plotTrajectory3D(to5DUVdS(trajp, lmb), xc=2, yc=3, zc=4, t=tp, linewidth=4, name=geonum, color='rgb' + str(color))


statplotterUV.save("{}/{}_UV_mu{}_lmb{}".format(folder, fname, mu, str(lmb).replace(".", ",")), dpi=500)
statplotterxU.save("{}/{}_xU_mu{}_lmb{}".format(folder, fname, mu, str(lmb).replace(".", ",")), dpi=500)

dynplotterxyU.export_html("{}/{}_x-y-U_mu{}_lmb{}.html".format(folder, fname, mu, lmb))
dynplotterxyU.export_pdf("{}/{}_x-y-U_mu{}_lmb{}.pdf".format(folder, fname, mu, lmb), title=False)

dynplotterZ0Z2Z4.export_html("{}/{}_Z0-Z2-Z4_mu{}_lmb{}.html".format(folder, fname, mu, lmb))
dynplotterZ0Z2Z4.export_pdf("{}/{}_Z0-Z2-Z4_mu{}_lmb{}.pdf".format(folder, fname, mu, lmb), title=False)

dynplotterZ2Z3Z4.export_html("{}/{}_Z2-Z3-Z4_mu{}_lmb{}.html".format(folder, fname, mu, lmb))
dynplotterZ2Z3Z4.export_pdf("{}/{}_Z2-Z3-Z4_mu{}_lmb{}.pdf".format(folder, fname, mu, lmb), title=False)

