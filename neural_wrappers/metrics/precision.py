import numpy as np
from .metric import Metric
from scipy.special import softmax

# @brief The thresholded variant of F1Score (not argmax, but rather correct and higher than some threshold value). To
#  be used mostly in corroboration with MetricThresholder Callback.
class ThresholdPrecision(Metric):
	def __init__(self):
		super().__init__("max")

	@staticmethod
	def classPrecision(results, labels):
		TP = np.logical_and(results==True, labels==True).sum()
		FP = np.logical_and(results==True, labels==False).sum()

		return TP / (TP + FP + 1e-5)

	def __call__(self, results, labels, threshold=0.5, **kwargs):
		numClasses = results.shape[-1]
		results = softmax(results, axis=-1)
		results = results >= threshold
		labels = labels.astype(np.bool)

		Precision = 0
		for i in range(numClasses):
			classPrecision = ThresholdPrecision.classPrecision(results[..., i], labels[..., i])
			if classPrecision == 0:
				numClasses -= 1
			Precision += classPrecision

		if numClasses > 0:
			Precision /= numClasses
		return Precision