import numpy as np

from grimpulsivewaves.plotting.dynamic import PlotlyDynamicPlotter

plotter = PlotlyDynamicPlotter()
plotter.plotHyperboloid(1)

plotter.show()
plotter.export_html("hyp_test.html", include_plotlyjs=True, include_mathjax=True)
print(plotter.fig)