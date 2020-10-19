from overrides import overrides
from typing import Callable, Dict
from .metric import Metric
from ..utilities import NWNumber, isBaseOf

# Small proxy metric that we can use if we don't provide any metric to wrap, just a callback.
class DefaultMetric(Metric):
	def __init__(self):
		super().__init__(direction="min")

	def __call__(self, results:NWNumber, labels:NWNumber, **kwargs):
		return results

class MetricWrapper(Metric):
	def __init__(self, callback:Callable[[NWNumber, NWNumber, Dict], NWNumber], wrappedMetric:Metric=DefaultMetric()):
		super().__init__(direction=wrappedMetric.direction)
		assert callable(callback)
		assert callable(wrappedMetric)
		assert not isBaseOf(callback, Metric)
		assert isBaseOf(wrappedMetric, Metric)
		self.callback = callback
		self.wrappedMetric = wrappedMetric

	@overrides
	def compareFunction(self, a:NWNumber, b:NWNumber) -> bool:
		return self.wrappedMetric.compareFunction(a, b)

	@overrides
	def getExtremes(self) -> Dict[str, NWNumber]:
		return self.wrappedMetric.getExtremes()

	@overrides
	def getDirection(self) -> str:
		return self.wrappedMetric.getDirection()

	@overrides
	def epochReduceFunction(self, results):
		return self.wrappedMetric.epochReduceFunction(results)

	@overrides
	def iterationReduceFunction(self, results):
		return self.wrappedMetric.iterationReduceFunction(results)

	@overrides
	def defaultValue(self):
		return self.wrappedMetric.defaultValue()

	# @brief The main method that must be implemented by a metric
	@overrides
	def __call__(self, results, labels, **kwargs):
		try:
			# self.callback should return the "result" of the metric
			# TODO: If I need to change this (delete default metric? Why did I put it in the first place)
			res = self.callback(results, labels, **kwargs)
			res2 = self.wrappedMetric(res, labels, **kwargs)
			return res2
		except Exception:
			breakpoint()
