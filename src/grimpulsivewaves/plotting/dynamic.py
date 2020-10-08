import numpy as np
import plotly.graph_objects as go

import random

class PlotlyDynamicPlotter:
    def __init__(self, labels=["x", "y", "z"], title="", aspectratio=None, xrange=None, yrange=None, zrange=None):
        layout = go.Layout(scene= dict(
            xaxis_title = labels[0],
            yaxis_title = labels[1],
            zaxis_title = labels[2]))
        self.fig = go.Figure(layout=layout)
        if xrange:
            self.fig.update_layout(scene=dict(
                xaxis = dict(range=xrange)
            ))
        if yrange:
            self.fig.update_layout(scene=dict(
                yaxis=dict(range=yrange)
            ))
        if zrange:
            self.fig.update_layout(scene=dict(
                zaxis=dict(range=zrange)
            ))
        #TODO: Set Z axis label
        self.fig.update_layout(showlegend=False, title=title)
        if aspectratio:
            self.fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=aspectratio[0], y=aspectratio[1], z=aspectratio[2]))

    def plotTrajectory3D(self, trajectory, color="#{:06x}".format(random.randint(0, 0xFFFFFF)).upper(), xc=1, yc=2, zc=3):
        xs = np.array([x[xc] for x in trajectory]).flatten()
        ys = np.array([x[yc] for x in trajectory]).flatten()
        zs = np.array([x[zc] for x in trajectory]).flatten()
        self.fig.add_scatter3d(x=xs, y=ys, z=zs, mode="lines", line=go.scatter3d.Line(color=color))

    def plotHyperboloid(self, l=1, color="#{:06x}".format(random.randint(0, 0xFFFFFF)).upper()):
        """
        This method has to be called before anything else is rendered (as it deletes everything
        already rendered)
        :param l: Cosmological constant
        """
        import plotly.figure_factory as ff #I hope Python is clever!
        from scipy.spatial import Delaunay

        eps = np.sign(l)
        a = np.sqrt(3/np.abs(l))
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(-1, 1, 20) #TODO: This should not be hard-coded

        u, v = np.meshgrid(u, v)

        u = u.flatten()
        v = v.flatten()

        if(eps > 0):
            x = a * np.cosh(v/a) * np.cos(u)
            y = a * np.cosh(v/a) * np.sin(u)
            z = a * np.sinh(v/a)
        else:
            x = a * np.cosh(v/a) * np.cos(u)
            z = a * np.cosh(v/a) * np.sin(u)
            y = a * np.sinh(v/a)

        points2D = np.vstack([u, v]).T

        tri = Delaunay(points2D)
        simplices = tri.simplices

        self.fig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, title="de Sitter",
                                aspectratio=dict(x=1, y=1, z=0.8), colomap=color)


    def show(self):
        self.fig.show()

    def export_html(self, path, include_plotlyjs=True):
        self.fig.write_html(path, include_plotlyjs=include_plotlyjs)