from grimpulsivewaves.coordinates import SchwarzschildCoordinates

from grimpulsivewaves.waves import Solution

from grimpulsivewaves.plotting import PlotlyDynamicPlotter

import numpy as np

rs = 1

def toCartNaive(x):
    return list(map(lambda x:[x[0],
                              x[1] * np.sin(x[2]) * np.cos(x[3]),
                              x[1] * np.sin(x[2]) * np.sin(x[3]),
                              x[1] * np.cos(x[2])], x))

def metric(x, u, rs):
    return -(1-rs/x[1]) * u[0] * u[0] + 1/(1-rs/x[1]) * u[1] * u[1] + x[1]*x[1] * (u[2] * u[2] + np.sin(x[2])**2 * u[3] * u[3])


initpos = [SchwarzschildCoordinates(np.array([0, 3, np.pi/2, 0])), SchwarzschildCoordinates(np.array([0, 6, np.pi/2, 0])), SchwarzschildCoordinates(np.array([0, 0.95, np.pi/2, 0]))]
initvel = [SchwarzschildCoordinates(np.array([1, 0, 0, 0]), dif=True), SchwarzschildCoordinates(np.array([1, 0, 0.052, 0]), dif=True), SchwarzschildCoordinates(np.array([0.1, 0.3, 0.5, 0]), dif=True)]

gen = Solution()

plotterRTphi = PlotlyDynamicPlotter(labels=["x", "z", "t"], aspectratio=[1, 1, 1], bgcolor="#ccffeb", fontsize=30)

for x0, u0n, geonum in zip(initpos, initvel, range(len(initvel))):
    norm = metric(x0, u0n, rs)
    u0 = (u0n / norm) if norm < 0 else u0n
    print(norm)
    pos, t = gen.generate_geodesic(x0, u0, (0, 400), christoffelParams=[rs], max_step=0.1)
    plotterRTphi.plotTrajectory3D(toCartNaive(pos), t=t, xc=1, yc=3, zc=0, color="rgb(" + str(255 - geonum * 80) + "," + str(geonum * 80) + ",0)")

plotterRTphi.export_html("schw.html")