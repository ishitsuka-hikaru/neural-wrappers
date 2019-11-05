from .callback import Callback
from ..pytorch.pytorch_utils import plotModelMetricHistory

class PlotMetrics(Callback):
	def __init__(self, metrics, plotBestBullet=None, dpi=120, **kwargs):
		super().__init__(**kwargs)
		assert len(metrics) > 0, "Expected a list of at least one metric which will be plotted."
		self.metrics = metrics
		self.dpi = dpi
		self.plotBestBullet = plotBestBullet
		if self.plotBestBullet == None:
			self.plotBestBullet = ["none"] * len(self.metrics)

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"]
		if not kwargs["isTraining"] or len(trainHistory) == 1:
			return

		for i, metric in enumerate(self.metrics):
			bullet = self.plotBestBullet[i]
			plotModelMetricHistory(metric, trainHistory, bullet)
