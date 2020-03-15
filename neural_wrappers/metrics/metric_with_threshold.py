import numpy as np
from .metric import Metric, Number

class MetricWithThreshold(Metric):
	def __init__(self, direction : str="min"):
		super().__init__(direction)

	def __call__(self, results : np.ndarray, labels : np.ndarray, threshold : np.ndarray=0.5, **kwargs) -> float:
		raise NotImplementedError("Should have implemented this")