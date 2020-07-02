from .callback import Callback
from .callback_name import CallbackName
from ..pytorch.utils import plotModelMetricHistory
from typing import List, Union, Tuple, Optional

class PlotMetrics(Callback):
	def __init__(self, metricNames : List[CallbackName], **kwargs):
		assert len(metricNames) > 0, "Expected a list of at least one metric which will be plotted."
		self.metricNames = metricNames
		self.directions = None
		super().__init__(**kwargs)

	def setup(self, model):
		if not self.directions is None:
			return

		metricDirections = []
		for metricName in self.metricNames:
			metric = model.getMetric(metricName)
			metricDirection = metric.getDirection()
			metricDirections.append(metricDirection)
		self.directions = metricDirections

	def onEpochEnd(self, **kwargs):
		self.setup(kwargs["model"])
		trainHistory = kwargs["trainHistory"]
		if not kwargs["isTraining"] or len(trainHistory) == 1:
			return

		for i in range(len(self.metricNames)):
			plotModelMetricHistory(trainHistory, self.metricNames[i], self.directions[i])

	def onCallbackSave(self, model):
		self.directions = None

	def __str__(self):
		assert len(self.metricNames) >= 1
		Str = str(self.metricNames[0])
		for i in range(len(self.metricNames)):
			Str += ", %s" % (str(self.metricNames[i]))
		return "PlotMetrics (%s)" % (Str)