from . import cli
from . import datasets
from . import evaluation
from . import models
from . import plotting
from .model_evaluator import ModelEvaluator
from .plotting.plot import plot
from .plotting.plot_redux import plot_redux
from .version import __version__, VERSION
from .analysis import Analyze

Evaluate = ModelEvaluator
Plot = plot
PlotRedux = plot_redux
