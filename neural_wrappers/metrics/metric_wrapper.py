from overrides import overrides
from typing import Callable, Dict, Union
from .metric import Metric
from ..callbacks import Callback
from ..utilities import NWNumber, isBaseOf

class MetricWrapper(Metric):
	def __init__(self, wrappedMetric:Callable[[NWNumber, NWNumber, Dict], NWNumber], direction:str="min"):
		assert not isBaseOf(wrappedMetric, Metric), "No need to wrap as it is already a metric/callback"
		self.wrappedMetric = wrappedMetric
		super().__init__(direction=direction)

	# @brief The main method that must be implemented by a metric
	@overrides
	def __call__(self, results, labels, **kwargs):
		try:
			res = self.wrappedMetric(results, labels, **kwargs)
			return res
		except Exception as e:
			print("\n____________________________________________\n%s" % str(e))
			print("Probably the metric '%s' doesn't have the **kwargs parameter defined on __call__" % self.name)
			breakpoint()

	def __str__(self):
		return "Metric Wrapper (%s)" % str(self.wrappedMetric)

	def __repr__(self):
		return str(self)