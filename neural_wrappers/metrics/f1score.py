import numpy as np
from .metric import Metric
from scipy.special import softmax

class F1Score(Metric):
	def __init__(self, categoricalLabels=True):
		super().__init__("max")
		self.categoricalLabels = categoricalLabels

	@staticmethod
	def classF1Score(results, labels):
		TP = np.logical_and(results==True, labels==True).sum()
		FP = np.logical_and(results==True, labels==False).sum()
		FN = np.logical_and(results==False, labels==True).sum()

		P = TP / (TP + FP + 1e-5)
		R = TP / (TP + FN + 1e-5)
		F1 = 2 * P * R / (P + R + 1e-5)
		return F1

	def __call__(self, results, labels, **kwargs):
		numClasses = results.shape[-1]
		argMaxResults = np.argmax(results, axis=-1)

		if self.categoricalLabels:
			argMaxLabels = np.argmax(labels, axis=-1)
		else:
			argMaxLabels = labels

		f1 = 0
		for i in range(numClasses):
			classF1 = F1Score.classF1Score(argMaxResults==i, argMaxLabels==i)
			if classF1 == 0:
				numClasses -= 1
			f1 += classF1
		if numClasses > 0:
			f1 /= numClasses

		return f1

# @brief The thresholded variant of F1Score (not argmax, but rather correct and higher than some threshold value). To
#  be used mostly in corroboration with MetricThresholder Callback.
class ThresholdF1Score(Metric):
	def __init__(self):
		super().__init__("max")

	def __call__(self, results, labels, threshold=0.5, **kwargs):
		numClasses = results.shape[-1]
		results = softmax(results, axis=-1)
		results = results >= threshold
		labels = labels.astype(np.bool)

		f1 = 0
		for i in range(numClasses):
			classF1 = F1Score.classF1Score(results[..., i], labels[..., i])
			if classF1 == 0:
				numClasses -= 1
			# print(i, results[..., i].shape, labels[..., i].shape, classF1)
			f1 += classF1

		if numClasses > 0:
			f1 /= numClasses
		return f1