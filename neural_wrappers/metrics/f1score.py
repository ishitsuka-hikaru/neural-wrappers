import numpy as np
from .metric_with_threshold import MetricWithThreshold
from .metric import Metric
from .precision import ThresholdPrecision
from .recall import ThresholdRecall

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-f1Score-and-f1Score-9250280bddc2
class ThresholdF1Score(MetricWithThreshold):
	def __init__(self):
		super().__init__("max")

	def computeF1Score(results: np.ndarray, labels : np.ndarray) -> np.ndarray:
		precision = ThresholdPrecision.computePrecision(results, labels)
		recall = ThresholdRecall.computeRecall(results, labels)
		f1Score = 2 * precision * recall / (precision + recall + 1e-8)
		return f1Score
		
	def __call__(self, results : np.ndarray, labels : np.ndarray, threshold : np.ndarray, **kwargs) -> float:
		results = np.uint8(results >= threshold)
		# Nans are used to specify classes with no labels for this batch
		f1Score = ThresholdF1Score.computeF1Score(results, labels)
		# Keep only position where f1Score is not nan.
		whereNotNaN = ~np.isnan(f1Score)
		f1Score = f1Score[whereNotNaN]
		# Mean over those valid classes.
		# return f1Score.mean()

		# It's better to compute the weighted mean of these predictions, instead of treating each element in this
		#  MB equally.
		whereOk = labels.sum(axis=0)
		whereOk = whereOk[whereNotNaN]
		return (f1Score * whereOk).sum() / whereOk.sum()

class F1Score(Metric):
	def __init__(self):
		super().__init__("max")
		self.thresholdF1Score = ThresholdF1Score()

	# @brief Since we don't care about a particular threshold, just to get the highest activation for each prediction,
	#  we can compute the max on the last axis (classes axis) and send this as threshold to the ThresholdAccuracy
	#  class.
	def __call__(self, results : np.ndarray, labels : np.ndarray, **kwargs) -> float:
		Max = results.max(axis=-1, keepdims=True)
		return self.thresholdF1Score(results, labels, Max)