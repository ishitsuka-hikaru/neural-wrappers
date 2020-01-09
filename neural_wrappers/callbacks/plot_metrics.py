from .callback import Callback
from ..pytorch.pytorch_utils import plotModelMetricHistory

class PlotMetrics(Callback):
	def __init__(self, metrics, **kwargs):
		super().__init__(**kwargs)
		assert len(metrics) > 0, "Expected a list of at least one metric which will be plotted."
		self.metrics = list(metrics)

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"]
		if not kwargs["isTraining"] or len(trainHistory) == 1:
			return

		for metric in self.metrics:
			direction = kwargs["model"].getMetrics()[metric].getDirection()
			plotModelMetricHistory(metric, trainHistory, direction)
