import numpy as np
from .metric import Metric
from ..utilities import NWNumber

class MetricWithThreshold(Metric):
	def __init__(self, direction : str="min"):
		super().__init__(direction)

	def __call__(self, results : NWNumber, labels : NWNumber, threshold : NWNumber, **kwargs) -> NWNumber: #type: ignore[override]
		raise NotImplementedError("Should have implemented this")