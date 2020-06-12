import numpy as np

from src.grimpulsivewaves.coordinates.coordinates import Cartesian

from src.grimpulsivewaves.waves import AichelburgSexlSolution

from src.grimpulsivewaves.plotting import StaticGeodesicPlotter

from src.grimpulsivewaves.integrators.geodesic_integrator import integrate_geodesic

wave = AichelburgSexlSolution(1) #Generate spacetime with wave


#INITIAL POSITION AND 4-VELOCITY
initCoords = Cartesian(np.array([0, 0, 1.2, 0.2]))
initCoords2 = Cartesian(np.array([0, 0, 0, 1]))
initVels = Cartesian(1. / np.sqrt((1 ** 2 + 0.1 ** 2)) * np.array([1, 0, 0.1, 0]), True)

initCoordsp, initVelsp = wave._refract(initCoords, initVels) #Refraction
initCoordsp2, initVelsp2 = wave._refract(initCoords2, initVels)
#TODO: use generate_geodesic from solution class (as in ring example)
#Consider implementig general wavefronts (coordinate transformations to shift it to U=0 would be necessary for
#computing refraction equations) and methods to find where geodesics pass wavefronts.


#Integration (not realy necessary)
tau0 = -0.2
refract = 0
tau1 = 0.2

#Integration, currently using 2 integrators (one to integrate back in time before crossing wavefront)
integrator1 = integrate_geodesic(initCoords, -initVels, (tau0, refract))
integrator2 = integrate_geodesic(initCoordsp, initVelsp, (refract, tau1))

integrator3 = integrate_geodesic(initCoords2, -initVels, (tau0, refract))
integrator4 = integrate_geodesic(initCoordsp2, initVelsp2, (tau0, refract))


#Integrated geodesic in M^-
geom = [Cartesian(x[4:], False) for x in integrator1.y.T]

geo2m = [Cartesian(x[4:], False) for x in integrator3.y.T]


#Integrated geodesic in M^+
geop = [Cartesian(x[4:], False) for x in integrator2.y.T]

geo2p = [Cartesian(x[4:], False) for x in integrator4.y.T]

#Init plotter
plotter = StaticGeodesicPlotter(use3d=True, zlabel="t", labels2d=["x", "z"])

#Plot
plotter.plot(geom, color="cyan", xc=2, yc=1, zc=0)
plotter.plot(geop, color="blue", xc=2, yc=1, zc=0)

plotter.plot(geo2m, color="magenta", xc=2, yc=1, zc=0)
plotter.plot(geo2p, color="Red", xc=2, yc=1, zc=0)

#Show and save plot
plotter.save(name="test.png", dpi=400)
plotter.show()
