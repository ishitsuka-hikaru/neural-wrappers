import numpy as np

class Metric:
	def __call__(self, results, labels, **kwargs):
		raise NotImplementedError("Should have implemented this")

class Accuracy(Metric):
	def __init__(self, categoricalLabels):
		self.categoricalLabels = categoricalLabels

	def __call__(self, results, labels, **kwargs):
		predicted = np.argmax(results, axis=1)
		labels = np.argmax(labels, axis=1) if self.categoricalLabels else labels
		correct = np.sum(predicted == labels)
		total = labels.shape[0]
		accuracy = correct / total * 100
		return accuracy