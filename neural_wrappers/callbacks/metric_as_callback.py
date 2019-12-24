from .callback import Callback
from ..metrics import Metric

# This class is used to convert metrics to callbacks which are called at each iteration. This is done so we unify
#  metrics and callbacks in one way. Stats and iteration messages can be computed for both cases thanks to this.
class MetricAsCallback(Callback):
	def __init__(self, metricName, metric):
		super().__init__(metricName)
		self.metric = metric

	# Returns "min" or "max" as defined in the class of each metric. If it's a function or some other type of metric
	#  that posed as MetricAsCallback and has no direction field defined, just ignore the exception and reutrn "min".
	def getDirection(self):
		try:
			return self.metric.direction
		except Exception:
			return "min"

	def onIterationEnd(self, results, labels, **kwargs):
		return self.metric(results, labels, **kwargs)

	def __call__(self, results, labels, **kwargs):
		return self.metric(results, labels, **kwargs)