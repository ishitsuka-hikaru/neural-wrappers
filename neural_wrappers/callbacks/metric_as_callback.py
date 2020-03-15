from .callback import Callback
from ..metrics import Metric
from typing import Optional

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

	# This is used by complex MetricAsCallbacks where we do some stateful computation at every iteration and we want
	#  to reduce it gracefully at the end of the epoch, so it can be stored in trainHistory, as well as for other
	#  callbacks to work nicely with it (SaveModels, PlotCallbacks, etc.). So, we apply a reduction function (default
	#  is identity, which might or might not work depending on algorithm).
	def reduceFunction(self, results):
		return results

	def onEpochEnd(self, **kwargs):
		return None

	def onEpochStart(self, **kwargs):
		return None

	def onIterationStart(self, **kwargs):
		return None

	def onIterationEnd(self, results, labels, **kwargs):
		return self.metric(results, labels, **kwargs)

	def __call__(self, results, labels, **kwargs):
		return self.onIterationEnd(results, labels, **kwargs)
