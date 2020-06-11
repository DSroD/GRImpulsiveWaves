import numpy as np

from src.grimpulsivewaves.coordinates.coordinates import Cartesian

from src.grimpulsivewaves.waves import AichelburgSexlSolution

from src.grimpulsivewaves.plotting import StaticGeodesicPlotter

from src.grimpulsivewaves.integrators.geodesic_integrator import integrate_geodesic

wave = AichelburgSexlSolution(4) #Generate spacetime with wave


#INITIAL POSITION AND 4-VELOCITY
initCoords = Cartesian(np.array([0, 1 / 2, 1 / 2, 0]))
initVels = Cartesian(1. / np.sqrt((1 + 10 ** 2 + 3 ** 2)) * np.array([1, 10, 0, 3]), True)

initCoordsp, initVelsp = wave._refract(initCoords, initVels) #Refraction
#TODO: Refraction could be done by integrator
#Consider implementig general wavefronts (coordinate transformations to shift it to U=0 would be necessary for
#computing refraction equations) and methods to find where geodesics pass wavefronts.


#Integration (not realy necessary)
tau0 = -0.2
refract = 0
tau1 = 0.2

#Integration, currently using 2 integrators (one to integrate back in time before crossing wavefront)
integrator1 = integrate_geodesic(initCoords, -initVels, (tau0, refract))
integrator2 = integrate_geodesic(initCoordsp, initVelsp, (refract, tau1))

#Init plotter
plotter = StaticGeodesicPlotter(use3d=True, zlabel="t")

#Integrated geodesic in M^-
geom = [Cartesian(x[4:], False) for x in integrator1.y.T]
#Plot
plotter.plot(geom, xc=1, yc=2, zc=0)

#Integrated geodesic in M^+
geop = [Cartesian(x[4:], False) for x in integrator2.y.T]
#Plot
plotter.plot(geop, xc=1, yc=2, zc=0)

#Show and save plot
plotter.show()
plotter.save("test.png", dpi=400)
