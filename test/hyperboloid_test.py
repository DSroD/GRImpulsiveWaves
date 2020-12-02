import numpy as np

from grimpulsivewaves.plotting.dynamic import PlotlyDynamicPlotter

plotter = PlotlyDynamicPlotter(aspectratio=[1, 1, 1], labels=["Z4", "Z1", "Z0"], xrange=[-5, 5], yrange=[-5, 5],
                               zrange=[-5, 5], showSpikes=True)

plotter.plotCutAndPasteHyperboloid(2, 1, (-4, 4), opacity=0.3, color="rgb(153,153,255)", drawImpulse=True, showlegend=True)

plotter.show()
plotter.export_html("hyp_test+.html", include_plotlyjs=True, include_mathjax=True)