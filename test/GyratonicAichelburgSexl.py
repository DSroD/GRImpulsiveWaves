import numpy as np
from src.grimpulsivewaves.coordinates.coordinates import NullTetradAichelburgSexlGyraton
from src.grimpulsivewaves.waves import AichelburgSexlGyratonSolution
from src.grimpulsivewaves.plotting import StaticGeodesicPlotter

import random


N = 10 #Number of geodesics
chi = np.pi/16 #Chi parameter

initpos = [NullTetradAichelburgSexlGyraton(np.array([0, 0, 5*-np.sin(phi) + 5j * np.cos(phi), -5*np.sin(phi) - 5j * np.cos(phi)]))
           for phi in np.linspace(0, 2*np.pi * (N-1.) / N, num=N)]
u0 = np.array([1+0j, 0+0j, 0+0j, 0+0j]) #Make it null geodesics
initvels = [NullTetradAichelburgSexlGyraton(u0 * np.linalg.norm(u0), True) for u in range(N)]

wave = AichelburgSexlGyratonSolution(0.2, chi) #Generate spacetime with wave

plotter = StaticGeodesicPlotter(use3d=True, labels2d=["x", "v"], zlabel="t") #Init 3D plotter

plotter3d2 = StaticGeodesicPlotter(use3d=True, labels2d=["x", "y"], zlabel="t") #Init 3D plotter

plotter2d = StaticGeodesicPlotter(use3d=False, labels2d=["$\mathcal{U}$", "$\mathcal{V}$"]) #Init 2D plotter

#For each init pos generate geodesic (splitted)
for x0, v0 in zip(initpos, initvels):
    trajm, trajp = wave.generate_geodesic(x0, v0, (-0.5, 0.8), christoffelParams=[chi])
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()

    # TODO: Create ToCartesian function because using so many lambdas here is just bad practice

    plotter.plot(list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]),
                                     1./np.sqrt(2.) * np.real(x[0] - x[1]),
                                     1./np.sqrt(2.) * np.real(x[2] + x[3]),
                                     1./np.sqrt(2.) * np.imag(x[2] - x[3])], trajm)), line="-", color=color, xc=2, yc=1, zc=0)

    plotter.plot(list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]),
                                     1./np.sqrt(2.) * np.real(x[0] - x[1]),
                                     1./np.sqrt(2.) * np.real(x[2] + x[3]),
                                     1./np.sqrt(2.) * np.imag(x[2] - x[3])], trajp)), color=color, xc=2, yc=1, zc=0)


    plotter3d2.plot(list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]),
                                        1./np.sqrt(2.) * np.real(x[0] - x[1]),
                                        1./np.sqrt(2.) * np.real(x[2] + x[3]),
                                        1./np.sqrt(2.) * np.imag(x[2] - x[3])], trajm)), line="-", color=color, xc=2, yc=3, zc=0)
    plotter3d2.plot(list(map(lambda x: [1./np.sqrt(2.) * np.real(x[0] + x[1]),
                                        1./np.sqrt(2.) * np.real(x[0] - x[1]),
                                        1./np.sqrt(2.) * np.real(x[2] + x[3]),
                                        1./np.sqrt(2.) * np.imag(x[2] - x[3])], trajp)), color=color, xc=2, yc=3, zc=0)

    # Plot U-V graph, we can see all the geodesics joined to one
    plotter2d.plot(trajm, color="black", line="-", xc=0, yc=1)
    plotter2d.plot(trajp, color="black", xc=0, yc=1)

#TODO: This test file might be good for planned feature to select one of the geodesics and create projection to it's 3-space for
#   all other geodesics. Time parameter will change to propper time of selected "observer" test particle.

plotter.save("3d_ring_gyra.png")
plotter.show()
plotter2d.save("2d_test_gyra.png")
plotter3d2.show()
plotter3d2.save("3d_ring2_gyra.png")
plotter2d.show()

