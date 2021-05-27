import numpy as np

from grimpulsivewaves.waves import GeneralLambdaGyratonSolution

from grimpulsivewaves.coordinates import DeSitterConstantHeavisideGyratonNullTetrad

from grimpulsivewaves.plotting import StaticGeodesicPlotter
from grimpulsivewaves.plotting import PlotlyDynamicPlotter

N = 20
r = 1

folder = "ads_gyraton"
filename = f"timelike__u_05__v_1__ring__r_{r}"

v0 = np.array([0.5, 1, 0, 0], dtype=np.complex128)

mu = 2
ch = 2
lmb = 1
chi = ch * np.pi

intrange = (-1, 3)

def _h(x, lmb, mu):
    return 1. / 24. * mu[0] * (2 * (-6 + x[2] * x[3] * lmb) + (6 + x[2] * x[3] * lmb) * np.log(6 * np.abs(x[2] * x[3] * lmb)))


# def _hz(self, x):
#    return self.mu * 2. * self.l * (-36. + x[2]**2 * x[3]**2 * self.l**2 - 12 * x[2] * x[3] * self.l *
#                                    np.log(6/(x[2] * x[3] * self.l)))/ (3. * x[2] *
#                                                                       (6. + x[2] * x[3] * self.l)**2)

def _hz(x, lmb, mu):
    return mu[0] / (24 * x[2]) * (-6 + x[2] * x[3] * lmb * (1 + np.log((6 * np.abs(lmb)) / (lmb * lmb * np.abs(x[2] * x[3])))))


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


def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2 * sigma**2))


def genRGB(i, n, add = 0):
    mean = n / 2.
    sigma = mean / 2.
    return [gaussian((i + add) % n, mean, sigma) * 255., 0., (1.-gaussian((i + add) % n, mean, sigma)) * 255.]

def omega(x, lmb):
    return 1 - 1./6. * lmb * np.real(x[2] * x[3] - x[0] * x[1])

def g(u, v, x, lmb):
    return (u[2] * v[3] + u[3] * v[2] - u[0] * v[1] - u[1] * v[0]) / omega(x, lmb)**2

initpos = [DeSitterConstantHeavisideGyratonNullTetrad(np.array([0, 0, r*np.exp(x*1j), r*np.exp(-x*1j)])) for x in np.linspace(0, 2 * np.pi * (N-1)/N, num=N)]
initvels = [DeSitterConstantHeavisideGyratonNullTetrad(-v0 / g(v0, v0, x0, lmb) , True) for x0 in initpos]

wave = GeneralLambdaGyratonSolution(lmb, chi, _h, _hz, mu)

static_plotter = StaticGeodesicPlotter(labels2d=["$U$", "$V$"])
dynamic_plotter = PlotlyDynamicPlotter(["x", "y", "U"], aspectratio=[1, 1, 1], showSpikes=False, bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)
dynamic_plotter2 = PlotlyDynamicPlotter(["x", "y", "V"], aspectratio=[1, 1, 1], showSpikes=False, bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)
dynamic_plotter3 = PlotlyDynamicPlotter(["x", "z", "t"], aspectratio=[1, 1, 1], showSpikes=False, bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)
dynamic_plotter4 = PlotlyDynamicPlotter(["x", "y", "z"], aspectratio=[1, 1, 1], showSpikes=False, bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)
dynamic_plotter5 = PlotlyDynamicPlotter(["x", "y", "t"], aspectratio=[1, 1, 1], showSpikes=False, bgcolor="#ccffeb", fontsize=30, ticks=True, tick_fontsize=15)

for num, geo in enumerate(zip(initpos, initvels)):
    x0 = geo[0]
    u0 = geo[1]

    a = wave.generate_geodesic(x0, u0, intrange, max_step=0.05, christoffelParams=[lmb, chi, False], christoffelParamsPlus=[lmb, chi, True])

    c = genRGB(num, N, 10)

    color = "rgb(" + str(c[0]) + "," + str(c[1]) + "," + str(c[2]) + ")"

    trajm0, trajp0 = a[0]
    tm0, tp0 = a[1]

    dynamic_plotter.plotTrajectory3D(toGyraUVxy(trajm0), color=color, xc=2, yc=3, zc=0,
                              name="Geod " + str(num) + " (-)", t=tm0, dash="longdashdot")
    dynamic_plotter.plotTrajectory3D(toGyraUVxy(trajp0), color=color, xc=2, yc=3, zc=0,
                              name="Geod " + str(num) + " (+)", t=tp0)

    dynamic_plotter2.plotTrajectory3D(toGyraUVxy(trajm0), color=color, xc=2, yc=3, zc=1,
                                     name="Geod " + str(num) + " (-)", t=tm0, dash="longdashdot")
    dynamic_plotter2.plotTrajectory3D(toGyraUVxy(trajp0), color=color, xc=2, yc=3, zc=1,
                                     name="Geod " + str(num) + " (+)", t=tp0)

    dynamic_plotter3.plotTrajectory3D(toGyraCart(trajm0), color=color, xc=2, yc=1, zc=0,
                                     name="Geod " + str(num) + " (-)", t=tm0, dash="longdashdot")
    dynamic_plotter3.plotTrajectory3D(toGyraCart(trajp0), color=color, xc=2, yc=1, zc=0,
                                     name="Geod " + str(num) + " (+)", t=tp0)

    dynamic_plotter4.plotTrajectory3D(toGyraCart(trajm0), color=color, xc=2, yc=3, zc=1,
                                     name="Geod " + str(num) + " (-)", t=tm0, dash="longdashdot")
    dynamic_plotter4.plotTrajectory3D(toGyraCart(trajp0), color=color, xc=2, yc=3, zc=1,
                                     name="Geod " + str(num) + " (+)", t=tp0)

    dynamic_plotter5.plotTrajectory3D(toGyraCart(trajm0), color=color, xc=2, yc=3, zc=0,
                                     name="Geod " + str(num) + " (-)", t=tm0, dash="longdashdot")
    dynamic_plotter5.plotTrajectory3D(toGyraCart(trajp0), color=color, xc=2, yc=3, zc=0,
                                     name="Geod " + str(num) + " (+)", t=tp0)




dynamic_plotter.export_html(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_uxy.html", include_plotlyjs=True, include_mathjax=True)
dynamic_plotter.export_pdf(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_uxy.pdf", eye=(2.5, 1.5, 0.6))

dynamic_plotter2.export_html(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_vxy.html", include_plotlyjs=True, include_mathjax=True)
dynamic_plotter2.export_pdf(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_vxy.pdf", eye=(2.5, 1.5, 0.6))

dynamic_plotter3.export_html(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_xzt.html", include_plotlyjs=True, include_mathjax=True)
dynamic_plotter3.export_pdf(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_xzt.pdf", eye=(2.5, 1.5, 0.6))

dynamic_plotter4.export_html(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_xyz.html", include_plotlyjs=True, include_mathjax=True)
dynamic_plotter4.export_pdf(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_xyz.pdf", eye=(2.5, 1.5, 0.6))

dynamic_plotter5.export_html(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_xyt.html", include_plotlyjs=True, include_mathjax=True)
dynamic_plotter5.export_pdf(f"{folder}/lmb_{lmb}/{filename}__mu_{mu}__chi_{ch}pi_xyt.pdf", eye=(2.5, 1.5, 0.6))