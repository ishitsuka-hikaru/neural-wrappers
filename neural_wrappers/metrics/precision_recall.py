import numpy as np
from .metric import Metric
from .precision import ThresholdPrecision
from .recall import ThresholdRecall

# @brief The thresholded variant of F1Score (not argmax, but rather correct and higher than some threshold value). To
#  be used mostly in corroboration with MetricThresholder Callback.
class ThresholdPrecisionRecall(Metric):
	def __init__(self):
		super().__init__("max")
		self.precisionMetric = ThresholdPrecision()
		self.recallMetric = ThresholdRecall()

	def __call__(self, results, labels, threshold=0.5, **kwargs):
		precision = self.precisionMetric(results, labels, threshold)
		recall = self.recallMetric(results, labels, threshold)
		return np.array([precision, recall])