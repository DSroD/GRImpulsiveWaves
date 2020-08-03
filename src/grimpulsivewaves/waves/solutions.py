import numpy as np

from ..integrators import integrate_geodesic

class Solution:

    def generate_geodesic(self, x0, v0, range, dim=4, christoffelParams=None, coordinateParams=None, max_step=np.inf):
        if x0.type != v0.type:
            raise ValueError("x0 and v0 has to be in same coordinate representation")

        sol = integrate_geodesic(x0, v0, min(range), max(range), christoffelParams, max_step)

        return [x0.coordinate_type(x[dim:]) for x in sol.y.T]


class RefractionSolution(Solution):

    def generate_geodesic(self, x0, v0, range, dim=4, splitted=True, christoffelParams=None, coordinateParams=None, max_step=np.inf):
        """

        :param x0: Initial particle 4-position
        :param v0: Initial particle 4-velocity
        :return: List of trajectories (each trajectory is list of respective coordinates)
        """

        # TODO: This has to be done using metric, line element should be added to coords or wave lol (also null geodesics luls at me)
        #if np.dot(v0.x, v0.x) != 1:
            #raise ValueError("v0 should be normed to 1")

        if x0.type != v0.type:
            raise ValueError("x0 and v0 has to be in same coordinate representation")

        # TODO: Following checks are not general enough (conversion to classical null tetrad might not be always possible)

        #if x0.type == "null_tetrad" and x0[0] != 0:
            #raise ValueError("x0 has to lie on wavefront")

        #elif x0.to_nulltetrad()[0] != 0:
            #raise ValueError("x0 has to lie on wavefront")

        xp, vp = self._refract(x0, v0)

        solminus = integrate_geodesic(x0, -v0, (min(range), 0), christoffelParams, max_step)
        solplus = integrate_geodesic(xp, vp, (0, max(range)), christoffelParams, max_step)


        #TODO: Return afinne parameter as list aswell (as propper time of each particle if massive)

        if splitted:
            trajectories = []
            trajectories.append([x0.coordinate_type(x[dim:]) for x in solminus.y.T[:-1]])
            trajectories.append([x0.coordinate_type(x[dim:]) for x in solplus.y.T])
            return trajectories
        else:
            return [x0.coordinate_type(x[dim:]) for x in np.append(solminus.y.T[:-1], solplus.y.T)]


class AichelburgSexlSolution(RefractionSolution):
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
                            _x[1] - self._h(_x),
                            _x[2], _x[3]])
            _dhz = self._hz(_x)
            _nu = np.array([_u[0],
                            _u[1] + _dhz * _u[2] + np.conj(_dhz) * _u[3] + _dhz * np.conj(_dhz) * _u[0],
                            _u[2] + np.conj(_dhz) * _u[0],
                            _u[3] + _dhz * _u[0]])
        else:
            raise Exception("Something went wrong while converting to internal coordinate representation")


        #TODO: Do this more pythonic
        if keepCoordinates and _x.type != x.type:
            _x.x = _nx
            _u.x = _nu
            return x.coordinate_type.convert(_x), u.coordinate_type.convert(_u)
        else:
            return _x.coordinate_type(_nx), _x.coordinate_type(_nu, True)

    def _h(self, x):
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if x.type == "null_tetrad":
            #Branch cut [-inf , 0]
            return -self.mu * np.log(2 * x[2] * x[3])
        if x.type == "null_cartesian":
            return -self.mu * np.log(x[2]**2 + x[3]**2)

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



class HottaTanakaSolution(RefractionSolution):
    #TODO: this :)
    def __init__(self, l, mu):
        """
        Hotta - Tanaka solution is generalized Aichelburg-Sexsl solution for non-zero cosmological
        constant. This solution implements 5D formalism.
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

    def _h(self, x):
        raise NotImplementedError()

    def _hz(self, x):
        raise NotImplementedError()

class AichelburgSexlGyratonSolution(RefractionSolution):
    #TODO: finish this class I guess
    def __init__(self, mu, chi):
        """
        Frolov-Fursaev Gyraton is an explicit family of waves with twisting source. This solution implements Aichelburg-Sexl
        solution with additional off-diagonal terms in spacetime in front of the wavefront.
        :param mu: Mu parameter of wave
        :param chi: Twisting parameter of spacetime u > 0
        """
        if chi == 0:
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

        # Several coordinate representations of refraction equations are presented
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if not u.dif:
            raise Exception("4-velocity argument u has to be differential")
        defined = ["aichelburg_sexl_gyraton_null_tetrad"]
        if x.type not in defined:
            #TODO: Check if conversion exist first
            _x = x.to_aichelburg_sexl_gyraton_null_tetrad()
            _u = u.to_aichelburg_sexl_gyraton_null_tetrad()
        else:
            _x = x
            _u = u
        _nx = np.array([_x[0],
                        _x[1] - self._h(_x),
                        _x[2], _x[3]])
        _dhz = self._hz(_x)
        _nu = np.array([_u[0],
                        _u[1] + _dhz * _u[2] + np.conj(_dhz) * _u[3] + (_dhz * np.conj(_dhz) - self.chi**2 / (4. * _x[2] * _x[3])) * _u[0],
                        _u[2] + (np.conj(_dhz) - 1j * self.chi / (2. * _x[3])) * _u[0],
                        _u[3] + (_dhz + 1j * self.chi / (2 * _x[2])) * _u[0]])

        # TODO: Do this more pythonic
        if keepCoordinates and _x.type != x.type:
            _x.x = _nx
            _u.x = _nu
            return x.coordinate_type.convert(_x), u.coordinate_type.convert(_u)
        else:
            return _x.coordinate_type(_nx), _x.coordinate_type(_nu, True)


    def _h(self, x):
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if x.type == "aichelburg_sexl_gyraton_null_tetrad":
            # Branch cut [-inf , 0]
            return -self.mu * np.log(2 * x[2] * x[3])
        else:
            raise Exception("Error in inner conversion to Aichelburg Sexl Gyratonic coordinate system")

    def _hz(self, x):
        if x.dif:
            raise Exception("Coordinate argument x cannot be differential")
        if x.type == "aichelburg_sexl_gyraton_null_tetrad":
            # Branch cut [-inf , 0]
            return -self.mu * 1. / x[2]
        else:
            raise Exception("Error in inner conversion to Aichelburg Sexl Gyratonic coordinate system")

    

class FrolovFursaevGyratonLambda(RefractionSolution):
    def __init__(self, mu, chi, l):
        """

        :param mu:
        :param chi:
        :param l:
        """