import numpy as np

class Coordinates:
    def __init__(self, x, dif):
        self.x = x
        self.dif = dif

    def __getitem__(self, item):
        return self.x[item]

    def __setitem__(self, key, value):
        self.x[key] = value

    def __neg__(self):
        return self.coordinate_type(-self.x, self.dif)

    #TODO: Are these methods necessary? General coordinate system might not have conversion to cartesian etc
    def to_cartesian(self):
        raise NotImplementedError()

    def to_nulltetrad(self):
        raise NotImplementedError()

    def to_nullcartesian(self):
        raise NotImplementedError()

    @property
    def coordinate_type(self):
        return Coordinates


class Cartesian(Coordinates):
    def __init__(self, x, dif=False):
        """
        Cartesian coordinates on Minkowski metric ds^2 = -dt^2 + dz^2 + dx^2 + dy^2
        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)
        self.type = "cartesian"

    @staticmethod
    def christoffel(x):
        return np.zeros((4, 4, 4))

    def to_nulltetrad(self):
        if self.dif:
            return NullTetrad(np.array([1./np.sqrt(2) * (self.x[0] - self.x[1]),
                                        1./np.sqrt(2) * (self.x[0] + self.x[1]),
                                        1./np.sqrt(2) * (self.x[2] - 1j * self.x[3]),
                                        1./np.sqrt(2) * (self.x[2] + 1j * self.x[3])]),
                              True)
        else:
            return NullTetrad(np.array([1./np.sqrt(2) * (self.x[0] - self.x[1]),
                                        1./np.sqrt(2) * (self.x[0] + self.x[1]),
                                        1./np.sqrt(2) * (self.x[2] + 1j * self.x[3]),
                                        1./np.sqrt(2) * (self.x[2] - 1j * self.x[3])]))

    def to_spherical(self):
        raise NotImplementedError()

    def to_null_polar(self):
        raise NotImplementedError()

    @staticmethod
    def convert(x):
        return x.to_cartesian()

    @property
    def coordinate_type(self):
        return Cartesian

class GyratonicCartesian(Coordinates):
    #TODO: Finish this class
    def __init__(self, x, dif=False):
        """
        Cartesian coordinates on Minkowski metric with additional 2 Theta(z-t)J(x, y, z-t) dx dy term
        (Theta is Heaviside function)
        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)
        self.type = "gyratonic_cartesian"
    @staticmethod
    def christoffel(x):
        if(x[0] - x[1] <= 0):
            return np.zeros((4, 4, 4))
        else:
            raise NotImplementedError()

class NullTetrad(Coordinates):
    def __init__(self, x, dif=False):
        """

        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)
        self.type = "null_tetrad"

    def to_cartesian(self):
        if self.dif:
            return Cartesian(np.array([1. / np.sqrt(2) * np.real(self.x[0] + self.x[1]),
                                       1. / np.sqrt(2) * np.real(self.x[0] - self.x[1]),
                                       1. / np.sqrt(2) * np.real(self.x[2] + self.x[3]),
                                       1. / np.sqrt(2) * np.imag(self.x[3] - self.x[2])]),
                             True)
        else:
            return Cartesian(np.array([1. / np.sqrt(2) * np.real(self.x[0] + self.x[1]),
                                       1. / np.sqrt(2) * np.real(self.x[0] - self.x[1]),
                                       1. / np.sqrt(2) * np.real(self.x[2] + self.x[3]),
                                       1. / np.sqrt(2) * np.imag(self.x[2] - self.x[3])]))

    def to_nullcartesian(self):
        if self.dif:
            return NullCartesian(np.array([self.x[0], self.x[1],
                                           1. / np.sqrt(2) * np.real(self.x[2] + self.x[3]),
                                           1. / np.sqrt(2) * np.imag(self.x[3] - self.x[2])]),
                                 True)
        else:
            return NullCartesian(np.array([self.x[0], self.x[1],
                                           1. / np.sqrt(2) * np.real(self.x[2] + self.x[3]),
                                           1. / np.sqrt(2) * np.imag(self.x[2] - self.x[3])]))

    @staticmethod
    def convert(x):
        return x.to_nulltetrad()

    @property
    def coordinate_type(self):
        return NullTetrad

    @staticmethod
    def christoffel(x):
        return np.zeros((4, 4, 4))

class GyratonicNullTetrad(Coordinates):
    #TODO: Finish this class
    def __init__(self, x, dif=False):
        """

        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)
        self.type = "general_gyratonic_null_tetrad"

class NullCartesian(Coordinates):
    def __init__(self, x, dif=False):
        """

        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)
        self.type = "null_cartesian"

    def to_cartesian(self):
        return Cartesian(np.array([1. / np.sqrt(2) * (self.x[0] + self.x[1]),
                                   1. / np.sqrt(2) * (self.x[0] - self.x[1]),
                                   self.x[2], self.x[3]]), self.dif)

    def to_nulltetrad(self):
        if self.dif:
            return NullTetrad(np.array([self.x[0], self.x[1],
                                        1. / np.sqrt(2) * (self.x[2] - 1j * self.x[3]),
                                        1. / np.sqrt(2) * (self.x[2] + 1j * self.x[3])]), True)
        else:
            return NullTetrad(np.array([self.x[0], self.x[1],
                                        1. / np.sqrt(2) * (self.x[2] + 1j * self.x[3]),
                                        1. / np.sqrt(2) * (self.x[2] - 1j * self.x[3])]))

    @staticmethod
    def convert(x):
        return x.to_nullcartesian()

    @property
    def coordinate_type(self):
        return NullCartesian

    @staticmethod
    def christoffel(x):
        return np.zeros((4, 4, 4))

class NullTetradFrolovFursaev(Coordinates):
    def __init__(self, x, chi, dif=False):
        """
        Gyratonic Null Tetrad Coordinates for Frolov-Fursaev gyraton spacetime
        :param x:
        :param dif:
        """
        super().__init__(x, dif)
        self.type = "frolov_fusarev_tetrad"
        self.chi = chi

    @staticmethod
    def christoffel(x, chi):
        if (x[3] - x[0] <= 0):
            return np.zeros((4, 4, 4))
        else:
            ch = np.zeros((4, 4, 4))
            ch[2, 1, 2] = -1.0j * chi / (2 * x[2] * x[2])
            ch[2, 2, 1] = -1.0j * chi / (2 * x[2] * x[2])
            ch[2, 1, 3] = -1.0j * chi / (2 * x[3] * x[3])
            ch[2, 3, 1] = -1.0j * chi / (2 * x[3] * x[3])

    @staticmethod
    def convert(x):
        return x.to_nulltetradfrolovfursaev()

    @property
    def coordinate_type(self):
        return NullTetradFrolovFursaev


class Spherical(Coordinates):
    def __init__(self, x, dif=False):
        """
        -dt^2 + dr^2 +r^2 d\theta^2 + r^2 sin^2 \theta d\phi^2
        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)
        self.type = "spherical"

    @staticmethod
    def christoffel(x):
        cf = np.zeros((4, 4, 4))
        cf[1, 2, 2] = - x[1]  # r theta theta
        cf[1, 3, 3] = - x[1] * np.sin(x[2])**2  # r phi phi
        cf[2, 1, 2] = 1. / x[1]
        cf[2, 2, 1] = 1. / x[1]
        cf[2, 3, 3] = - np.sin(x[2]) * np.cos(x[2])
        cf[3, 1, 3] = 1. / x[1]
        cf[3, 3, 1] = 1. / x[1]
        cf[3, 2, 3] = np.cos(x[2]) / np.sin(x[2])
        cf[3, 3, 2] = np.cos(x[2]) / np.sin(x[2])
        return cf


    def to_cartesian(self):
        pass

    def to_null(self):
        pass

    def to_nullpolar(self):
        if self.dif:
            return NullPolar(np.array([1./np.sqrt(2) * (self.x[0] - self.x[1]),
                                        1./np.sqrt(2) * (self.x[0] + self.x[1]),
                                        self.x[2],
                                        self.x[3]]))
        else:
            return NullPolar(np.array([]))

    @staticmethod
    def convert(x):
        return x.to_spherical()

    @property
    def coordinate_type(self):
        return Spherical


class Polar(Coordinates):
    def __init__(self, x, dif=False):
        """

        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)

class NullPolar(Coordinates):
    def __init__(self, x, dif=False):
        """

        :param x: Numpy array of numbers
        :param dif: True if this is velocities, default false if coordinates
        """
        super().__init__(x, dif)
        self.type = "nullpolar"

    def to_cartesian(self):
        pass

    def to_null(self):
        pass

    def to_spherical(self):
        pass

    @staticmethod
    def Convert(x):
        return x.to_nullpolar()

    @property
    def coordinate_type(self):
        return NullPolar


class FiveDimensional(Coordinates):
    def __init__(self, x, l, dif=False):
        super().__init__(x, dif)
        self.lmb = l
        self.a = np.sqrt(3/np.abs(l))
