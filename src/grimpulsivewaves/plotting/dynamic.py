import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.io import write_image
import random

class PlotlyDynamicPlotter:
    def __init__(self, labels=["x", "y", "z"], title="", aspectratio=None, xrange=None, yrange=None, zrange=None, showSpikes=True, spikeColor="#000000", bgcolor="#fff"):
        self.labels = labels

        axis = dict(showline=True,
                    linewidth=4,
                    title=dict(font=dict(size=20)),
                    showticklabels=False,
                    backgroundcolor=bgcolor)

        layout = go.Layout(scene= dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2],
            xaxis=axis,
            yaxis=axis,
            zaxis=axis)
            )

        self.fig = go.Figure(layout=layout)
        if xrange:
            self.fig.update_layout(scene=dict(
                xaxis=dict(range=xrange)
            ))
        if yrange:
            self.fig.update_layout(scene=dict(
                yaxis=dict(range=yrange)
            ))
        if zrange:
            self.fig.update_layout(scene=dict(
                zaxis=dict(range=zrange)
            ))
        if showSpikes:
            self.fig.update_layout(scene=dict(
                xaxis=dict(spikecolor=spikeColor,
                           spikesides=False,
                           spikethickness=2),
                yaxis=dict(spikecolor=spikeColor,
                           spikesides=False,
                           spikethickness=2),
                zaxis=dict(spikecolor=spikeColor,
                           spikesides=False,
                           spikethickness=2)
            ))
        else:
            self.fig.update_layout(scene=dict(
                xaxis=dict(showspikes=False),
                yaxis=dict(showspikes=False),
                zaxis=dict(showspikes=False)
            ))
        self.fig.update_layout(title=dict(text=title, font=dict(size=35), x=0.5, xanchor="center", y=0.8, yanchor="top"))
        if aspectratio:
            self.fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=aspectratio[0], y=aspectratio[1], z=aspectratio[2]))

    def plotTrajectory3D(self, trajectory, color="#{:06x}".format(random.randint(0, 0xFFFFFF)).upper(), xc=1, yc=2, zc=3, name="", t=None, showlegend=False):
        hinfo = "%{fullData.name}<br><br>" + self.labels[0] + ": %{x}<br>" + self.labels[1] + ": %{y}<br>" + self.labels[2] + ": %{z}<br><br>" + r"tau: %{text}<extra></extra>"

        xs = np.array([x[xc] for x in trajectory]).flatten()
        ys = np.array([x[yc] for x in trajectory]).flatten()
        zs = np.array([x[zc] for x in trajectory]).flatten()

        self.fig.add_scatter3d(x=xs, y=ys, z=zs, mode="lines", line=go.scatter3d.Line(color=color), name=name,
                               hoverinfo='all')
        if t is not None:
            self.fig['data'][-1].update(text=t)

        self.fig['data'][-1].update(hovertemplate=hinfo)

        self.fig['data'][-1].update(showlegend=showlegend)




    def plotHyperboloid(self, l=1, vsize=(-2,2), opacity=0.5, plot_edges=False,  color="rgb(" + str(random.randint(50,100)) + "," + str(random.randint(50,100)) + "," + str(random.randint(50,100)) + ")", drawImpulse=False, showlegend=False):
        """
        Generate hyperboloid
        :param l: Cosmological constant
        """
        import plotly.figure_factory as ff #I hope Python is clever!
        from scipy.spatial import Delaunay

        eps = np.sign(l)
        a = np.sqrt(3/np.abs(l))
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(vsize[0], vsize[1], 60) #TODO: resolution should not be hard-coded

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

        _tempfig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, show_colorbar=False, colormap=color, plot_edges=plot_edges,
                                     aspectratio=dict(x=1, y=1, z=0.8))
        _tempfig['data'][0].update(opacity=opacity)
        _tempfig['data'][0].update(hoverinfo='skip')
        _tempfig['data'][0].update(hoverinfo='skip')
        _tempfig['data'][0].update(name="Hyperboloid")
        _tempfig['data'][0].update(showlegend=showlegend)
        if drawImpulse:
            v = np.linspace(vsize[0], vsize[1], 10)
            z = v[:-1]
            y = v[:-1]
            x = 10 * [-a]
            x2 = 10 * [a]

            _tempfig.add_scatter3d(x=x, y=y, z=z, mode="lines", line=go.scatter3d.Line(color="black", width=4), name="U = infinity", hoverinfo='skip', showlegend=showlegend, opacity=0.4)
            _tempfig.add_scatter3d(x=x2, y=y, z=z, mode="lines", line=go.scatter3d.Line(color="black", width=8), name="U = 0", hoverinfo='skip', showlegend=showlegend, opacity=0.8)

        self.fig.add_traces(_tempfig.data)

    def plotCutAndPasteHyperboloid(self, H, l, vsize=(-2, 2), opacity=0.5, plot_edges=False,  color="rgb(" + str(random.randint(50,100)) + "," + str(random.randint(50,100)) + "," + str(random.randint(50,100)) + ")", drawImpulse=False, showlegend=False):
        from scipy.spatial import Delaunay

        a = np.sqrt(3 / np.abs(l))
        eps = np.sign(l)
        uo = np.linspace(0, 2 * np.pi, 700)
        v = np.linspace(vsize[0], vsize[1], 500)  # TODO: resolution should not be hard-coded

        uo, v = np.meshgrid(uo, v)

        u = uo.flatten()
        v = v.flatten()
        #TODO: OPRAVIT, ZKUST https://stackoverflow.com/questions/25060103/determine-sum-of-numpy-array-while-excluding-certain-values
        if (eps > 0):
            x = a * np.cosh(v / a) * np.cos(u)
            z = a * np.sinh(v / a)
            y = a * np.cosh(v / a) * np.sin(u)

        else:
            x = a * np.cosh(v/a) * np.cos(u)
            z = a * np.cosh(v / a) * np.sin(u)
            y = a * np.sinh(v / a)

        xp = np.array([a if c - b >= 0 else np.nan for a, b, c in zip(x, y, z)]).reshape(uo.shape)
        xm = np.array([a if c - b <= 0 else np.nan for a, b, c in zip(x, y, z)]).reshape(uo.shape)
        yp = np.array([b - 1. / np.sqrt(2) * H if c - b >= 0 else np.nan for a, b, c in zip(x, y, z)]).reshape(uo.shape)
        ym = np.array([b if c - b <= 0 else np.nan for a, b, c in zip(x, y, z)]).reshape(uo.shape)
        zp = np.array([c + 1. / np.sqrt(2) * H if c - b >= 0 else np.nan for a, b, c in zip(x, y, z)]).reshape(uo.shape)
        zm = np.array([c if c - b <= 0 else np.nan for a, b, c in zip(x, y, z)]).reshape(uo.shape)

        _tempfig = go.Figure(data=[
            go.Surface(x=xp, y=yp, z=zp, colorscale=[color, color]),
            go.Surface(x=xm, y=ym, z=zm, colorscale=[color, color])
        ])
        _tempfig['data'][0].update(name="Hyperboloid +")
        _tempfig['data'][1].update(name="Hyperboloid -")
        _tempfig.update_traces(showscale=False, opacity=opacity, hoverinfo='skip', showlegend=showlegend)
        if drawImpulse:
            v = np.linspace(vsize[0], vsize[1], 10)
            z = v[:-1]
            y = v[:-1]
            x = 10 * [-a]
            x2 = 10 * [a]

            _tempfig.add_scatter3d(x=x, y=y, z=z, mode="lines", line=go.scatter3d.Line(color="black", width=4),
                                   name="U- = infinity", hoverinfo='skip', showlegend=showlegend , opacity=0.4)
            _tempfig.add_scatter3d(x=x2, y=y, z=z, mode="lines", line=go.scatter3d.Line(color="black", width=8),
                                   name="U- = 0", hoverinfo='skip', showlegend=showlegend, opacity=0.8)
            _tempfig.add_scatter3d(x=x , y=y - 1. / np.sqrt(2) * H, z=z + 1. / np.sqrt(2) * H, mode="lines", line=go.scatter3d.Line(color="black", width=4),
                                   name="U+ = infinity", hoverinfo='skip', showlegend=showlegend, opacity=0.4)
            _tempfig.add_scatter3d(x=x2, y=y - 1. / np.sqrt(2) * H, z=z + 1. / np.sqrt(2) * H, mode="lines", line=go.scatter3d.Line(color="black", width=8),
                                   name="U+ = 0", hoverinfo='skip', showlegend=showlegend, opacity=0.8)

        self.fig.add_traces(_tempfig.data)




    def show(self):
        self.fig.show()

    def export_html(self, path, include_plotlyjs=True, include_mathjax=False):
        if include_mathjax:
            include_mathjax = 'cdn'
        self.fig.write_html(path, include_plotlyjs=include_plotlyjs, include_mathjax=include_mathjax)


    #TODO: Do something with default camera
    def export_pdf(self, path, eye=(1.25, 1.25, 1.25), up=(.0, .0, 1.0), orbit=False, title=True):
        """
        This requires Kaleido, install using "pip install -U Kaleido".
        :param path: Path of resulting file
        :param eye: Eye of camera
        :return:
        """
        camera = dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
            up=dict(x=up[0], y=up[1], z=up[2])
        )
        if orbit:
            self.fig.update_layout(scene = dict(dragmode='orbit'))

        self.fig.update_layout(scene_camera=camera, showlegend=False)
        write_image(self.fig, path, format="pdf", scale=3, engine="kaleido", width=1024, height=1024)
        self.fig.update_layout(scene_camera=dict(eye=dict(x=1.25, y=1.25, z=1.25), up=dict(x=.0,y=.0,z=1.0)), showlegend=True, scene=dict(dragmode='turntable'))