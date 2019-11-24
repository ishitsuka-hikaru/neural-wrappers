from .callback import Callback

# This class is used to convert metrics to callbacks which are called at each iteration. This is done so we unify
#  metrics and callbacks in one way. Stats and iteration messages can be computed for both cases thanks to this.
class MetricAsCallback(Callback):
	def __init__(self, metricName, metric):
		super().__init__(metricName)
		self.metric = metric

	def onIterationEnd(self, results, labels, **kwargs):
		return self.metric(results, labels, **kwargs)

	def __call__(self, results, labels, **kwargs):
		return self.metric(results, labels, **kwargs)