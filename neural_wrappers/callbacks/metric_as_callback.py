import numpy as np
from typing import List, Union, Dict, Optional
from .callback import Callback
from ..metrics import Metric
from ..utilities import NWNumber

# This class is used to convert metrics to callbacks which are called at each iteration. This is done so we unify
#  metrics and callbacks in one way. Stats and iteration messages can be computed for both cases thanks to this.
# Another use case of this class is to enable more complex metrics.
class MetricAsCallback(Callback):
	def __init__(self, metricName : str, metric : Optional[Metric] = None):
		super().__init__(metricName)
		self.metric = metric

	# Returns "min" or "max" as defined in the class of each metric. If it's a function or some other type of metric
	#  that posed as MetricAsCallback and has no direction field defined, just ignore the exception and reutrn "min".
	def getDirection(self) -> str:
		try:
			return self.metric.getDirection()
		except Exception:
			return "min"

	def epochReduceFunction(self, results) -> NWNumber:
		return results

	def iterationReduceFunction(self, results):
		return results

	def defaultValue(self) -> NWNumber:
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
