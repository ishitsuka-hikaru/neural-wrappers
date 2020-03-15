import numpy as np
from .metric_as_callback import MetricAsCallback
from ..metrics import MetricWithThreshold
from typing import List, Union

Number = Union[float, int]

# This wraps a metric with threshold by providing (a variable numober of) thresholds to be tested against
# Results are stored at each iteration and the epoch results are saved as the running mean of results.
class MetricWithThresholdWrapper(MetricAsCallback):
	def __init__(self, metricName : str, metric : MetricWithThreshold, \
		thresholds : Union[Number, np.ndarray, List[Number]] ):
		super().__init__(metricName, metric)
		self.thresholds = thresholds

	def onIterationEnd(self, results, labels, **kwargs):
		if type(self.thresholds) in (float, int):
			return self.metric(results, labels, threshold=self.thresholds, **kwargs)
		else:
			return np.array([self.metric(results, labels, threshold=x, **kwargs) for x in self.thresholds])