import numpy as np
from .metric import Metric
from scipy.special import softmax

# @brief The thresholded variant of F1Score (not argmax, but rather correct and higher than some threshold value). To
#  be used mostly in corroboration with MetricThresholder Callback.
class ThresholdRecall(Metric):
	def __init__(self):
		super().__init__("max")

	@staticmethod
	def classRecall(results, labels):
		TP = np.logical_and(results==True, labels==True).sum()
		FN = np.logical_and(results==False, labels==True).sum()
		R = TP / (TP + FN + 1e-5)
		return R

	def __call__(self, results, labels, threshold=0.5, **kwargs):
		numClasses = results.shape[-1]
		results = softmax(results, axis=-1)
		results = results >= threshold
		labels = labels.astype(np.bool)

		Recall = 0
		for i in range(numClasses):
			classRecall = ThresholdRecall.classRecall(results[..., i], labels[..., i])
			if classRecall == 0:
				numClasses -= 1
			Recall += classRecall

		if numClasses > 0:
			Recall /= numClasses
		return Recall