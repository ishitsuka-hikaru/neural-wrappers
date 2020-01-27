import numpy as np
from functools import reduce

class Metric:
	# @param[in] direction Defines the "direction" of the metric, as in if the better value means it is minimized or
	#  maximized. For example, Loss functions (or errors in general) are minimized, thus "min". However, other metrics
	#  such as Accuracy or F1Score are to be maximized, hence "max". Defaults to "min".
	def __init__(self, direction="min"):
		assert direction in ("min", "max")
		self.direction = direction

	def __call__(self, results, labels, **kwargs):
		raise NotImplementedError("Should have implemented this")

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