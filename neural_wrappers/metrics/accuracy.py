import numpy as np
from functools import reduce
from .metric import Metric
from scipy.special import softmax

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

# @brief The thresholded variant of Accuracy (not argmax, but rather correct and higher than some threshold value). To
#  be used mostly in corroboration with MetricThresholder Callback.
class ThresholdAccuracy(Metric):
	def __init__(self):
		super().__init__("max")

	def __call__(self, results, labels, threshold=0.5, **kwargs):
		results = softmax(results, axis=-1)
		results = results[labels == 1]
		results = results >= threshold
		return results.mean()