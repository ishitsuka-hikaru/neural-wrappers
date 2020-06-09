import numpy as np
from .metric_with_threshold import MetricWithThreshold
from .metric import Metric
from ..utilities import NWNumber

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-recall-and-recall-9250280bddc2
class ThresholdRecall(MetricWithThreshold):
	def __init__(self):
		super().__init__("max")

	def computeRecall(results: np.ndarray, labels : np.ndarray) -> np.ndarray:
		TP = results * labels
		FN = (1 - results) * labels
		TP = TP.sum(axis=0)
		FN = FN.sum(axis=0)
		res = (TP / (TP + FN + 1e-8))

		# We only care about those results that have actual labels (so we don't get 0 recall for a class that has
		#  no valid prediction in this MB). We mask the irelevant classes with nans.
		whereOk = (labels.sum(axis=0) > 0).astype(np.float32)
		whereOk[whereOk == 0] = np.nan

		res = res * whereOk
		return res

	def __call__(self, results : np.ndarray, labels : np.ndarray, threshold : np.ndarray, **kwargs) -> NWNumber: #type: ignore[override]
		results = np.uint8(results >= threshold)
		# Nans are used to specify classes with no labels for this batch
		recall = ThresholdRecall.computeRecall(results, labels)
		# Keep only position where recall is not nan.
		whereNotNaN = ~np.isnan(recall)
		recall = recall[whereNotNaN]
		# Mean over those valid classes.
		# return recall.mean()

		# It's better to compute the weighted mean of these predictions, instead of treating each element in this
		#  MB equally.
		whereOk = labels.sum(axis=0)
		whereOk = whereOk[whereNotNaN]
		return (recall * whereOk).sum() / whereOk.sum()

class Recall(Metric):
	def __init__(self):
		super().__init__("max")
		self.thresholdRecall = ThresholdRecall()

	# @brief Since we don't care about a particular threshold, just to get the highest activation for each prediction,
	#  we can compute the max on the last axis (classes axis) and send this as threshold to the ThresholdAccuracy
	#  class.
	def __call__(self, results : np.ndarray, labels : np.ndarray, **kwargs): #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		return self.thresholdRecall(results, labels, Max)
