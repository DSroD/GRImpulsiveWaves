import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


class StaticGeodesicPlotter:
    def __init__(self, ax=None, use3d=False, figsize=(8, 8), labels2d=["x", "y"], zlabel = "z"):
        self.ax = ax
        self.use3d = use3d
        if not self.ax:
            self.fig, self.ax = plt.subplots(figsize=figsize)
            self.ax.set_aspect(1) #Just to make sure ratio is 1:1
            if self.use3d:
                self.ax = plt.axes(projection="3d")
                self.ax.set_zlabel(zlabel)
            self.ax.set_xlabel(labels2d[0])
            self.ax.set_ylabel(labels2d[1])

    def _euclid_dist(self, x, y, z=0):
        return np.sqrt(x*x + y*y + z*z)


    def _set_scaling(self, x_range, y_range, z_range, lim):
        if x_range < lim and y_range < lim and z_range < lim:
            return
        if x_range < lim:
            self.ax.set_xlim([-lim, lim])
        if y_range < lim:
            self.ax.set_ylim([-lim, lim])
        if z_range < lim:
            self.ax.set_zlim([-lim, lim])

    def plot(self, trajectory, line="--", color="#{:06x}".format(random.randint(0, 0xFFFFFF)).upper(), xc=1, yc=2, zc=3):
        xs = np.array([x[xc] for x in trajectory])
        ys = np.array([x[yc] for x in trajectory])
        zs = np.array([x[zc] for x in trajectory])
        if not self.use3d:
            self.ax.plot(xs, ys, line, color=color)
        else:
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            z_range = max(zs) - min(zs)
            self._set_scaling(x_range, y_range, z_range, 1)
            self.ax.plot(xs, ys, zs, line, color=color)


    def show(self):
        plt.show()

    def save(self, name="geodesics.png", dpi=300):
        plt.savefig(name, dpi=dpi)
