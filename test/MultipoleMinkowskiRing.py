import numpy as np

from grimpulsivewaves.coordinates import NullTetrad

from grimpulsivewaves.waves import GeneralMinkowskiRefractionSolution

from grimpulsivewaves.plotting import PlotlyDynamicPlotter
from grimpulsivewaves.plotting import StaticGeodesicPlotter

def r(phi):
    # return 2 - 2 * np.sin(phi) + np.sin(phi) * np.sqrt(np.abs(np.cos(phi))) / (np.sin(phi) + 1.4) # Shape of heart
    # return 4 + np.cos(phi) + np.sin(6*phi) + np.cos(2*phi + 1) # "Random shape"
    # Circle
    return 1

def H(x, args):
    a = 1./np.sqrt(2) * np.real(x[2])
    b = 1./np.sqrt(2) * np.imag(x[2])
    phi = np.angle(x[2])
    r = np.sqrt(a**2 + b**2)
    if r < 0.03:
        return np.NaN
    h = - args[0] * np.log(2 * x[2] * x[3]) + args[1] * 1./r * np.cos(phi)
    return np.real(h)

def H_z(x, args):
    #phi = np.angle(x[2])
    r = np.real(2 * x[2] * x[3])
    if r < 0.01:
        return np.NaN
    return - args[0] / x[2] - args[1] * np.sqrt(2)/(x[2]**2)


N = 30 #Number of geodesics
r2 = 2
mu = np.array([0])
bu1 = np.array([0, 0.2, 0.4])
col = ["g", "r", "teal"]

xran = 4
yran = 4
zran = [-2, 3]

u0 = [np.array([0.5, 1, 0, 0], dtype=np.complex128) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
initvels = [NullTetrad(x, dif=True) for x in u0]

initpos = [NullTetrad(np.array([0, 0, r(phi) * np.exp(1j * phi), r(phi) * np.exp(- 1j * phi)], dtype=np.complex128))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]

#initpos = [NullTetrad(np.array([0, 0, x + 1j * y, x - 1j*y], dtype=np.complex128))
#          for x in np.linspace(-1, 1, num=N) for y in np.linspace(-1, 1, num=N)]

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
plotName = "Multipole/DipoleTimelike"


for b1 in bu1:
    staticplotter = StaticGeodesicPlotter(labels2d=[r"$\mathcal{U}$", r"$\mathcal{V}$"], labelsize=16, aspect='auto',
                                          figsize=(6, 2.1))
    for m, i in zip(mu, range(len(mu))):
        wave = GeneralMinkowskiRefractionSolution(H, H_z, m, b1)
        plotter = PlotlyDynamicPlotter(title="",
                                       aspectratio=[1, 1, 1], labels=lab,
                                       xrange=[-xran, xran], yrange=[-yran, yran], zrange=zran,
                                       bgcolor="#ccffeb", fontsize=30)

        staticplotter2 = StaticGeodesicPlotter(labels2d=["x", "z"], labelsize=30, aspect='auto',
                                              figsize=(12, 6))
        staticplotter2.ax.plot([0, 0], [-5, 5], "k-")

        for x0, u0, geonum in zip(initpos, initvels, range(0, len(initpos))):

            a = wave.generate_geodesic(x0, u0, (-3, 6), max_step=0.2, verbose=False)

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


        plotter.export_html("{}_{}{}{}_b0={}_b1={}.html".format(plotName, lab[0], lab[1], lab[2], m, b1), include_plotlyjs=True, include_mathjax=True)
        plotter.export_pdf("{}_{}{}{}_b0={}_b1={}.pdf".format(plotName, lab[0], lab[1], lab[2], m, b1), eye=(0, 0, 2.5), up=(0., 0., 1.))
        staticplotter2.ax.set_ylim([-0.6, 2.5])
        staticplotter2.save("{}_xz_b0={}_b1={}.png".format(plotName, m, b1), dpi=700, showlegend=False)

    staticplotter.save("{}_UV_b1={}.png".format(plotName, b1), dpi=600, showlegend=True)

funcplotter = PlotlyDynamicPlotter(title="",
                                       aspectratio=[1, 1, 1], labels=lab,
                                       xrange=[-2.2, 2.2], yrange=[-2.2, 2.2], zrange=[-10, 10],
                                       bgcolor="#ccffeb", fontsize=30)

funcplotterre = PlotlyDynamicPlotter(title="",
                                       aspectratio=[1, 1, 1], labels=lab,
                                       xrange=[-2.2, 2.2], yrange=[-2.2, 2.2], zrange=[-10, 10],
                                       bgcolor="#ccffeb", fontsize=30)

funcplotterim = PlotlyDynamicPlotter(title="",
                                       aspectratio=[1, 1, 1], labels=lab,
                                       xrange=[-2.2, 2.2], yrange=[-2.2, 2.2], zrange=[-10, 10],
                                       bgcolor="#ccffeb", fontsize=30)

funcplotter2 = PlotlyDynamicPlotter(title="",
                                       aspectratio=[1, 1, 1], labels=lab,
                                       xrange=[-2.2, 2.2], yrange=[-2.2, 2.2], zrange=[-4, 4],
                                       bgcolor="#ccffeb", fontsize=30)


funcplotterre.plotSurface(lambda x, y, args: np.real(H_z(np.array([0, 0, x, y]), args)), 0, -1, xdomain=[-2, 2], ydomain=[-2, 2], xstep=0.01, ystep=0.01, complexNull=True, showlegend=True, name="Real", color="rgb(255,71,71)", color2="rgb(255, 249, 71)")
funcplotterim.plotSurface((lambda x, y, args: np.imag(H_z(np.array([0, 0, x, y]), args))), 0, -1, xdomain=[-2, 2], ydomain=[-2, 2], xstep=0.01, ystep=0.01, complexNull=True, showlegend=True, name="Imaginary", color="rgb(138, 202, 255)", color2="rgb(192, 46, 255)")
funcplotter.export_html("{}_FunctionH_z.html".format(plotName), include_plotlyjs=True)
funcplotterre.export_pdf("{}_FunctionH_zRe.pdf".format(plotName), eye=(0.5, 1.4, 1.25))
funcplotterim.export_pdf("{}_FunctionH_zIm.pdf".format(plotName), eye=(0.5, 1.4, 1.25))

funcplotter2.plotSurface(lambda x, y, args: H(np.array([0, 0, x, y]), args), 0, -1, xdomain=[-2, 2], ydomain=[-2, 2], xstep=0.05, ystep=0.05, complexNull=True, showlegend=True, name="Real", color="rgb(255,71,71)", color2="rgb(255, 249, 71)")
funcplotter2.export_html("{}_FunctionH.html".format(plotName), include_plotlyjs=True)
funcplotter2.export_pdf("{}_FunctionH.pdf".format(plotName), eye=(0.5, 1.4, 1.25))