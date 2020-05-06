from .metric import Metric
from typing import Callable, Dict
from ..utilities import NWNumber

class MetricWrapper(Metric):
	def __init__(self, callback : Callable[[NWNumber, NWNumber, Dict], NWNumber], \
		direction : str = "min"):
		super().__init__(direction)
		self.callback = callback

	def __call__(self, results : NWNumber, labels : NWNumber, **kwargs):
		return self.callback(results, labels, **kwargs)