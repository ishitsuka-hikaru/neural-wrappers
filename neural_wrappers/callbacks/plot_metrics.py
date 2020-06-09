from .callback import Callback
from ..pytorch.utils import plotModelMetricHistory
from typing import List, Union, Tuple, Optional

class PlotMetrics(Callback):
	def __init__(self, metricNames : List[Union[str, Tuple[str]]], directions : Optional[List[str]]=None, **kwargs):
		assert len(metricNames) > 0, "Expected a list of at least one metric which will be plotted."
		self.directions = ["min"] * len(metricNames)
		for i in range(len(metricNames)):
			if type(metricNames[i]) == str:
				metricNames[i] = (metricNames[i], )
			if not directions is None:
				self.directions[i] = directions[i]
		self.metricNames = metricNames
		super().__init__(**kwargs)

	def onEpochEnd(self, **kwargs):
		trainHistory = kwargs["trainHistory"]
		if not kwargs["isTraining"] or len(trainHistory) == 1:
			return

		for i in range(len(self.directions)):
			plotModelMetricHistory(trainHistory, self.metricNames[i], self.directions[i])

	def __str__(self):
		assert len(self.metricNames) >= 1
		Str = str(self.metricNames[0])
		for i in range(len(self.metricNames)):
			Str += ", %s" % (str(self.metricNames[i]))

		return "PlotMetrics (%s)" % (Str)