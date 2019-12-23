import numpy as np
from functools import reduce

class Metric:
	def __call__(self, results, labels, **kwargs):
		raise NotImplementedError("Should have implemented this")

class Accuracy(Metric):
	def __init__(self, categoricalLabels=True):
		self.categoricalLabels = categoricalLabels

	def __call__(self, results, labels, **kwargs):
		predicted = np.argmax(results, axis=-1)
		labels = np.argmax(labels, axis=-1) if self.categoricalLabels else labels
		total = reduce(lambda x, y: x*y, predicted.shape)
		correct = np.sum(predicted == labels)
		accuracy = correct / total * 100
		return accuracy

class F1Score(Metric):
	def __init__(self, categoricalLabels=True):
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