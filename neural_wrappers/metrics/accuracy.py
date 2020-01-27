import numpy as np
from functools import reduce
from .metric import Metric

class Accuracy(Metric):
	def __init__(self, categoricalLabels=True):
		super().__init__("max")
		self.categoricalLabels = categoricalLabels

	def __call__(self, results, labels, **kwargs):
		predicted = np.argmax(results, axis=-1)
		labels = np.argmax(labels, axis=-1) if self.categoricalLabels else labels
		# MBxHxWx3xNumClasses outputs => product of MB*H*W*3 (as NC are gone due to argmax above)
		total = reduce(lambda x, y: x*y, predicted.shape)
		correct = np.sum(predicted == labels)
		accuracy = correct / total * 100
		return accuracy
