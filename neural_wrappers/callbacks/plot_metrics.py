from .callback import Callback
from ..pytorch.utils import plotModelMetricHistory

class PlotMetrics(Callback):
	def __init__(self, metrics, **kwargs):
		assert len(metrics) > 0, "Expected a list of at least one metric which will be plotted."
		self.metrics = list(metrics)
		super().__init__(**kwargs)

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"]
		if not kwargs["isTraining"] or len(trainHistory) == 1:
			return

		for metric in self.metrics:
			direction = kwargs["model"].getMetrics()[metric].getDirection()
			plotModelMetricHistory(metric, trainHistory, direction)

	def __str__(self):
		assert len(self.metrics) >= 1
		Str = str(self.metrics[0])
		for i in range(len(self.metrics)):
			Str += ", %s" % (str(self.metrics[i]))

		return "PlotMetrics (%s)" % (Str)