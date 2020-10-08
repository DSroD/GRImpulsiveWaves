import numpy as np
from grimpulsivewaves.coordinates.coordinates import Cartesian
from grimpulsivewaves.waves import AichelburgSexlSolution
from grimpulsivewaves.plotting import StaticGeodesicPlotter

import random


N = 100 #Number of geodesics

initpos = [Cartesian(np.array([0, 0, -np.sin(phi), np.cos(phi)])) for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
u0 = np.array([-1, 1, 0, 0]) #Make it null geodesics
initvels = [Cartesian(u0 / np.linalg.norm(u0), True) for u in range(N)] #Can be generalized to different initial 4-vels

wave = AichelburgSexlSolution(0.6) #Generate spacetime with wave

plotter = StaticGeodesicPlotter(use3d=True, labels2d=["x", "z"], zlabel="t") #Init 3D plotter

plotter3d2 = StaticGeodesicPlotter(use3d=True, labels2d=["x", "y"], zlabel="t") #Init 3D plotter

plotter2d = StaticGeodesicPlotter(use3d=False, labels2d=["$\mathcal{U}$", "$\mathcal{V}$"]) #Init 2D plotter

#For each init pos generate geodesic (splitted)
for x0, u0 in zip(initpos, initvels):
    trajm, trajp = wave.generate_geodesic(x0, u0, (-0.2, 0.2))
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()

    plotter.plot(trajm, line="-", color=color, xc=2, yc=1, zc=0)
    plotter.plot(trajp, color=color, xc=2, yc=1, zc=0)


    plotter3d2.plot(trajm, line="-", color=color, xc=2, yc=3, zc=0)
    plotter3d2.plot(trajp, color=color, xc=2, yc=3, zc=0)

    # Plot U-V graph, we can see all the geodesics joined to one
    plotter2d.plot(list(map(lambda x: x.to_nulltetrad(), trajm)), color="black", line="-", xc=0, yc=1)
    plotter2d.plot(list(map(lambda x: x.to_nulltetrad(), trajp)), color="black", xc=0, yc=1)

#TODO: This test file might be good for planned feature to select one of the geodesics and create projection to it's 3-space for
#all other geodesics. Time parameter will change to propper time of selected "observer" test particle.

plotter.save("3d_ring.png")
plotter.show()
plotter2d.save("2d_test.png")
plotter3d2.show()
plotter3d2.save("3d_ring2.png")
plotter2d.show()



