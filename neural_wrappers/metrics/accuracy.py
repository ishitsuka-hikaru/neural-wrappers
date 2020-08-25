import numpy as np
from scipy.special import softmax
from overrides import overrides
from typing import Dict
from .metric_with_threshold import MetricWithThreshold
from .metric import Metric
from ..utilities import NWNumber

class ThresholdAccuracy(MetricWithThreshold):
	def __init__(self):
		super().__init__(direction="max")

	# @brief results and labels are two arrays of shape: (MB, N, C), where N is a variable shape (images, or just
	#  numbers), while C is the number of classes. The number of classes must coincide to both cases. We assume that
	#  the labels are one-hot encoded (1 on correct class, 0 on others), while we make no assumption about the results.
	#  This means that it can be before or after softmax or any other function. However, we also receive a threshold
	#  against which we compare this result and extract the "activations" (results > threshold, since it's a
	#  maximinzing metric). Then, we can get a bunch of such activations for each result, but we're only interested
	#  in the one that corresponds to the correct class, as said by label (result[labels == 1] == 1?). We sum those
	#  and divide by the number of items to get the thresholded accuracy.
	def __call__(self, results : np.ndarray, labels : np.ndarray, threshold : np.ndarray, **kwargs) -> float: #type: ignore[override]
		results = results >= threshold
		whereCorrect = labels == 1
		results = results[whereCorrect]
		return results.mean()

class Accuracy(Metric):
	def __init__(self):
		super().__init__(direction="max")
		self.thresholdAccuracy = ThresholdAccuracy()

	@overrides
	def getExtremes(self) -> Dict[str, NWNumber]:
		return {"min" : 0, "max" : 1}

	# @brief Since we don't care about a particular threshold, just to get the highest activation for each prediction,
	#  we can compute the max on the last axis (classes axis) and send this as threshold to the ThresholdAccuracy
	#  class.
	def __call__(self, results : np.ndarray, labels : np.ndarray, **kwargs) -> float: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		return self.thresholdAccuracy(results, labels, Max)

class ThresholdSoftmaxAccuracy(ThresholdAccuracy):
	def __call__(self, results : np.ndarray, labels : np.ndarray, threshold : np.ndarray, **kwargs) -> float: #type: ignore[override]
		results = softmax(results, axis=-1)
		return super().__call__(results, labels, threshold)