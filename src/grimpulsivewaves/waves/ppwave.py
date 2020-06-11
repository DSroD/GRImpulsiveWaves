import numpy as np

from ..integrators import integrate_geodesic

class Solution:

    def generate_geodesic(self, x0, v0, range, dim=4, splitted=True):
        """

        :param x0: Initial particle 4-position
        :param v0: Initial particle 4-velocity
        :return: List of trajectories
        """

        if np.dot(v0, v0) != 1:
            raise ValueError("v0 should be normed to 1")

        if x0.to_null_tetrad()[0] != 0:
            raise ValueError("x0 has to lie on wavefront")

        if x0.type != v0.type:
            raise ValueError("x0 and v0 has to be in same coordinate representation")

        zminus = np.append(v0.x, x0.x)
        xp, vp = self._refract(x0, v0)
        zplus = np.append(vp.x, xp.x)

        solminus = integrate_geodesic(zminus, (min(range), 0))
        solplus = integrate_geodesic(zplus, (0, max(range)))

        trajectories = []

        if splitted:
            trajectories.append([x[dim:] for x in solminus.y.T])
            trajectories.append([x[dim:] for x in solplus.y.T])
        else:
            trajectories.append([x[dim:] for x in np.append(solminus.y.T, solplus.y.T)])


class AichelburgSexlSolution(Solution):
    def __init__(self, mu):
        """
        Aichelburg - Sexl solution is a solution to Einstein equations describing
        massless blackhole moving at the speed of light. This solution effectively
        represents planar wave on Minkowski background.

        :param mu: Mu parameter of wave
        """

        self.mu = mu

    def _refract(self, x, u, keepCoordinates=True):
        """
        Internal method for geodesic plotting
        :param x: Position in internal coordinates
        :param u: Velocity in internal coordinates
        :param keepCoordinates: If false returns x and u in internal coordinates (for example when there is no inverse)
        :return: Position and velocity after refraction
        """
        #Several coordinate representations of refraction equations are presented
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if not u.dif:
            raise Exception("4-velocity argument u has to be differential")
        defined = ["null_tetrad"]
        if x.type not in defined:
            _x = x.to_nulltetrad()
            _u = u.to_nulltetrad()
        else:
            _x = x
            _u = u
        #Checking in case more represenations are implemented
        if _x.type == "null_tetrad":
            _nx = np.array([_x[0],
                            _x[1] + self._h(_x),
                            _x[2], x[3]])
            _dhz = self._hz(_x)
            _nu = np.array([_u[0],
                            _u[1] + _dhz * _u[2] + np.conj(_dhz) * _u[3] + _dhz * np.conj(_dhz) * _u[0],
                            _u[2] + np.conj(_dhz) * _u[0],
                            _u[3] + _dhz * _u[0]])
        else:
            raise Exception("Something went wrong while converting to internal coordinate representation")

        if keepCoordinates and _x.type != x.type:
            return x.coordinate_type.convert(_x), u.coordinate_type.convert(_u)
        else:
            return _x, _u

    def _h(self, x):
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if x.type == "null_tetrad":
            #Branch cut [-inf , 0]
            return -self.mu * np.log(2 * x[2] * x[3])
        if x.type == "null_cartesian":
            return --self.mu * np.log(x[2]**2 + x[3]**2)

    def _hz(self, x):
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if x.type == "null_tetrad":
            # Branch cut [-inf , 0]
            return -self.mu * 1./x[2]
        else:
            raise Exception("Tried to call wrong derivative of H")

    def _hx(self, x):
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if x.type == "null_cartesian":
            # Branch cut [-inf , 0]
            return -self.mu * 2. * x[2] / (x[2]**2 + x[3]**2)

    def _hy(self, x):
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if x.type == "null_cartesian":
            # Branch cut [-inf , 0]
            return -self.mu * 2. * x[3] / (x[2]**2 + x[3]**2)



class HottaTanakaSolution(Solution):
    def __init__(self, l, mu):
        """
        Hotta - Tanaka solution is generalized Aichelburg - Sexsl solution for non-zero cosmological
        constant.
        :param l: Cosmological constant
        :param mu: Mu parameter of wave
        """
        if l==0:
            raise Exception("For l=0 please use AichelburgSexlSolution class")
        self.l = l
        self.mu = mu


    def _refract(self, x, u, keepCoordinates=True):
        """
        Internal method for geodesic plotting
        :param x: Position in internal coordinates
        :param u: Velocity in internal coordinates
        :param keepCoordinates: If false returns x and u in internal coordinates (for example when there is no inverse)
        :return: Position and velocity after refraction
        """
        raise NotImplementedError()


class FrolovFursaevGyraton(Solution):
    #TODO: Better name for this class - it is only the "key example" of FF family (CARE NOT TO BREAK COMPATIBILITY!)
    def __init__(self, mu, chi):
        """
        Frolov-Fursaev Gyraton is an explicit family of waves with twisting
        :param mu: Mu parameter of wave
        :param chi: Twisting parameter of spacetime u > 0
        """
        if chi==0:
            raise Exception("For xi=0 please use AichelburgSexlSolution class")
        self.mu = mu
        self.chi = chi


    def _refract(self, x, u, keepCoordinates=True):
        """
        Internal method for geodesic plotting
        :param x: Position in internal coordinates
        :param u: Velocity in internal coordinates
        :param keepCoordinates: If false returns x and u in internal coordinates (for example when there is no inverse)
        :return: Position and velocity after refraction
        """
        raise NotImplementedError()

class FrolovFursaevGyratonLambda(Solution):
    def __init__(self, mu, chi, l):
        """

        :param mu:
        :param chi:
        :param l:
        """