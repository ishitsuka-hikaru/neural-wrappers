import numpy as np
from typing import List, Union, Dict, Optional
from .callback import Callback
from .callback_name import CallbackName
from ..metrics import Metric
from ..utilities import NWNumber

# This class is used to convert metrics to callbacks which are called at each iteration. This is done so we unify
#  metrics and callbacks in one way. Stats and iteration messages can be computed for both cases thanks to this.
# Another use case of this class is to enable more complex metrics.
class MetricAsCallback(Callback):
	def __init__(self, metricName:CallbackName, metric:Metric):
		super().__init__(metricName)
		self.metric = metric

	def getDirection(self) -> str:
		return self.metric.getDirection()

	def epochReduceFunction(self, results) -> NWNumber:
		try:
			return self.metric.epochReduceFunction(results)
		except Exception:
			return results

	def iterationReduceFunction(self, results):
		try:
			return self.metric.iterationReduceFunction(results)
		except Exception:
			return results

	def defaultValue(self) -> NWNumber:
		try:
			return self.metric.defaultValue()
		except Exception:
			return 0

	def onEpochEnd(self, **kwargs):
		pass

	def onEpochStart(self, **kwargs):
		pass

	def onIterationStart(self, **kwargs):
		pass

	def onIterationEnd(self, results, labels, **kwargs):
		return self.metric(results, labels, **kwargs)

	def __call__(self, results, labels, **kwargs):
		return self.onIterationEnd(results, labels, **kwargs)
