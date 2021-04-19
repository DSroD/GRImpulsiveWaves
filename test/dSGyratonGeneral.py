import numpy as np

from grimpulsivewaves.waves import GeneralLambdaGyratonSolution
from grimpulsivewaves.waves import LambdaGeneralSolution

from grimpulsivewaves.coordinates import DeSitterConstantHeavisideGyratonNullTetrad
from grimpulsivewaves.coordinates import DeSitterNullTetrad

from grimpulsivewaves.plotting import PlotlyDynamicPlotter


#HELPER FUNKCE

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
    return list(map(lambda x: [1./np.sqrt(2.) * np.real(x[1] + x[0] + thetaU(x) * h0) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.real(x[1] - x[0] - thetaU(x) * h0) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.real(x[2] + x[3]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               1./np.sqrt(2.) * np.imag(x[3] - x[2]) / (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3]))),
                               np.sqrt(3./(eps * lmb)) * (1 + lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))/
                               (1 - lmb/6. * (np.real(x[0]*x[1]) - np.real(x[2]*x[3])))], _a))[1:] #Ommit first so there is no line in the middle of the cut


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2 * sigma**2))

def genRGB(i, n, add = 0):
    mean = n / 2.
    sigma = mean / 2.
    return [gaussian((i + add) % n, mean, sigma) * 255., 0., (1.-gaussian((i + add) % n, mean, sigma)) * 255.]


# NASTAVENÍ

N = 25

mu = -1.
lmb = -1.
ch = 1.
chi = ch * np.pi

plot = [2, 3, 0]
lab = ["x", "y", "U"]
convertFunction = lambda x: toConformalUVxy(x)

plotHyperboloids = False

plotName = "dSConst"

initpos = [np.array([0., 0., np.exp(1j * phi), np.exp(-1j * phi)], dtype=np.complex128) for phi in np.linspace(0, 2 * np.pi * (N-1.) / N, num=N)]
#initpos = [np.array([0., 0., phi, phi], dtype=np.complex128) for phi in np.linspace(-2, 2, num=N)]
initvels = [0.3 * np.array([1., 0, 0., 0.], dtype=np.complex128) for phi in np.linspace(0, 2, num=N)]

# WAVEFRONT FUNKCE

def H1(x, l, args):
    return mu/24. * (-2. * (6 + x[2] * x[3] * l) + (6 - x[2] * x[3] * l) * np.log(6/(x[2] * x[3] * l)))
    #return 1

def H1Z(x, l, args):
    return - mu * 1./(24. * x[2]) * (6. + x[2] * x[3] * l + x[2] * x[3] * l * np.log(6./(x[2] * x[3] * l)))
    #return np.sqrt(np.abs(l)/3)

# ACTUAL CODE HERE PROCEED WITH CAUTION (please)

iposg = [DeSitterConstantHeavisideGyratonNullTetrad(x) for x in initpos]

ivel0 = [DeSitterNullTetrad(x, True) for x in initvels]
ivelg = [DeSitterConstantHeavisideGyratonNullTetrad(x, True) for x in initvels]

waveGENG = GeneralLambdaGyratonSolution(lmb, chi, H1, H1Z)
waveGEN0 = LambdaGeneralSolution(lmb, H1, H1Z)

plotter0 = PlotlyDynamicPlotter(title=r"",
                               aspectratio=[1, 1, 1], labels=lab,
                               xrange=[-5, 5], yrange=[-5, 5], zrange=[-5, 5], showSpikes=True, bgcolor="#d1f1ff")

plotterG = PlotlyDynamicPlotter(title=r"",
                               aspectratio=[1, 1, 1], labels=lab,
                               xrange=[-5, 5], yrange=[-5, 5], zrange=[-5, 5], showSpikes=True, bgcolor="#d1f1ff")

if plotHyperboloids:
    plotter0.plotHyperboloid(-3./2., (-5, 5), color="rgb(181,0,136)", drawImpulse=True, showlegend=True)
    plotterG.plotHyperboloid(-3./2., (-5, 5), color="rgb(181,0,136)", drawImpulse=True, showlegend=True)

for ipos, ivel, iposgyr, ivelgyr, geonum in zip(ipos0, ivel0, iposg, ivelg, range(N)):
    nogyr = waveGEN0.generate_geodesic(ipos, ivel, (-9, 9), max_step=0.2, christoffelParams=[lmb])

    gyr = waveGENG.generate_geodesic(iposgyr, ivelgyr, (-9, 9), max_step=0.2, christoffelParams=[lmb, chi, False],
                                     christoffelParamsPlus=[lmb, chi, True])

    c = genRGB(geonum, 2 * N, add=10)
    color = "rgb(" + str(c[0]) + "," + str(c[1]) + "," + str(c[2]) + ")"

    trajm0, trajp0 = nogyr[0]
    tm0, tp0 = nogyr[1]

    trajmg, trajpg = gyr[0]
    tmg, tpg = gyr[1]

    plotter0.plotTrajectory3D(convertFunction(trajm0), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geod " + str(geonum) + " (-)", t=tm0)
    plotter0.plotTrajectory3D(convertFunction(trajp0), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geod " + str(geonum) + " (+)", t=tp0)

    plotterG.plotTrajectory3D(convertFunction(trajmg), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geod " + str(geonum) + " (-)", t=tmg)
    plotterG.plotTrajectory3D(convertFunction(trajpg), color=color, xc=plot[0], yc=plot[1], zc=plot[2], name="Geod " + str(geonum) + " (+)", t=tpg)

plotter0.export_html(plotName + "NoGyra_" + lab[0] + lab[1] + lab[2] + ".html", include_plotlyjs=True, include_mathjax=True)
plotter0.export_pdf(plotName + "NoGyra_" + lab[0] + lab[1] + lab[2] + ".pdf", eye=(2.5, 0.5, 0.2))

plotterG.export_html(plotName + "Gyra_" + lab[0] + lab[1] + lab[2] + ".html", include_plotlyjs=True, include_mathjax=True)
plotterG.export_pdf(plotName + "Gyra_" + lab[0] + lab[1] + lab[2] + ".pdf", eye=(2.5, 0.5, 0.2))