import numpy as np
from src.grimpulsivewaves.coordinates.coordinates import Cartesian
from src.grimpulsivewaves.waves import AichelburgSexlSolution
from src.grimpulsivewaves.plotting import StaticGeodesicPlotter
from src.grimpulsivewaves.integrators.geodesic_integrator import integrate_geodesic

import random


N = 10 #Number of geodesics


initpos = [Cartesian(np.array([0, 0, 1.2* np.sin(phi), np.cos(phi)])) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
u0 = np.array([1, 0, 0, 0])
initvels = [Cartesian(u0 * np.linalg.norm(u0), True) for u in range(N)] #Can be generalized to different initial 4-vels

wave = AichelburgSexlSolution(0.6) #Generate spacetime with wave

plotter = StaticGeodesicPlotter(use3d=True, zlabel="t", labels2d=["x", "y"]) #Generate plotter

#For each init pos generate geodesic (splitted)

for x0, u0 in zip(initpos, initvels):
    trajm, trajp = wave.generate_geodesic(x0, u0, (-1, 0.2))
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()
    plotter.plot(trajm, color=color, xc=2, yc=3, zc=0)
    plotter.plot(trajp, color=color, xc=2, yc=3, zc=0)

#TODO: This test file might be good for planned feature to select one of the geodesics and create projection to it's 3-space for
#all other geodesics. Time parameter will change to propper time of selected "observer" test particle.

plotter.save("ring.png")
plotter.show()
