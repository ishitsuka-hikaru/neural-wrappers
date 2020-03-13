import numpy as np
from functools import reduce
from .metric import Metric
from scipy.special import softmax

class ThresholdAccuracy(Metric):
	def __init__(self):
		super().__init__("max")

	# @brief results and labels are two arrays of shape: (MB, N, C), where N is a variable shape (images, or just
	#  numbers), while C is the number of classes. The number of classes must coincide to both cases. We assume that
	#  the labels are one-hot encoded (1 on correct class, 0 on others), while we make no assumption about the results.
	#  This means that it can be before or after softmax or any other function. However, we also receive a threshold
	#  against which we compare this result and extract the "activations" (results > threshold, since it's a
	#  maximinzing metric). Then, we can get a bunch of such activations for each result, but we're only interested
	#  in the one that corresponds to the correct class, as said by label (result[labels == 1] == 1?). We sum those
	#  and divide by the number of items to get the thresholded accuracy.
	def __call__(self, results : np.ndarray, labels : np.ndarray, threshold : np.ndarray=0.5, **kwargs) -> float:
		results = results >= threshold
		whereCorrect = labels == 1
		results = results[whereCorrect]
		return results.mean()

class Accuracy(ThresholdAccuracy):
	# @brief Since we don't care about a particular threshold, just to get the highest activation for each prediction,
	#  we can compute the max on the last axis (classes axis) and send this as threshold to the ThresholdAccuracy
	#  class.
	def __call__(self, results : np.ndarray, labels : np.ndarray,**kwargs) -> float:
		Max = results.max(axis=-1, keepdims=True)
		return super().__call__(results, labels, Max)
