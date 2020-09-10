import numpy as np
from .metric_with_threshold import MetricWithThreshold
from .metric import Metric
from .precision import ThresholdPrecision
from .recall import ThresholdRecall

from ..utilities import npGetInfo

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-f1Score-and-f1Score-9250280bddc2
class ThresholdF1Score(MetricWithThreshold):
	def __init__(self):
		super().__init__("max")

	def computeF1Score(results: np.ndarray, labels : np.ndarray) -> np.ndarray:
		precision = ThresholdPrecision.computePrecision(results, labels)
		recall = ThresholdRecall.computeRecall(results, labels)
		f1Score = 2 * precision * recall / (precision + recall + 1e-8)
		return f1Score
		
	def __call__(self, results : np.ndarray, labels : np.ndarray, threshold : np.ndarray, **kwargs) -> float: #type: ignore[override]
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

# TODO
# class GlobalFScore(Metric):
# 	def __init__(self, threshold=0.5):
# 		super().__init__(direction="max")
# 		self.TP = np.array([0, 0], dtype=np.int64)
# 		self.FP = np.array([0, 0], dtype=np.int64)
# 		self.FN = np.array([0, 0], dtype=np.int64)
# 		self.threshold = threshold

# 	def epochReduceFunction(self, results):
# 		precision = self.TP / (self.TP + self.FP + 1e-5)
# 		recall = self.TP / (self.TP + self.FN + 1e-5)
# 		res = (precision * recall) / (precision + recall + 1e-5)
# 		# self.precision *= 0
# 		# self.recall *= 0
# 		print(res)
# 		print(res.mean())
# 		return res.mean()

# 	def iterationReduceFunction(self, results):
# 		precision = self.TP / (self.TP + self.FP + 1e-5)
# 		recall = self.TP / (self.TP + self.FN + 1e-5)
# 		return (precision * recall) / (precision + recall + 1e-5)

# 	def __call__(self, y, t, **k):
# 		yWhite = y[..., 0] >= self.threshold
# 		tWhite = t[..., 0] >= self.threshold
# 		yBlack = y[..., 0] < self.threshold
# 		tBlack = t[..., 0] < self.threshold

# 		TPWhite = yWhite * tWhite
# 		FPWhite = yWhite * (1 - tWhite)
# 		FNWhite = (1 - yWhite) * tWhite

# 		TPBlack = yBlack * tBlack
# 		FPBlack = yBlack * (1 - tBlack)
# 		FNBlack = (1 - yBlack) * tBlack

# 		self.TP += [TPWhite.sum(), TPBlack.sum()]
# 		self.FP += [FPWhite.sum(), FPBlack.sum()]
# 		self.FN += [FNWhite.sum(), FNBlack.sum()]
# 		return 0

class F1Score(Metric):
	def __init__(self):
		super().__init__("max")
		self.thresholdF1Score = ThresholdF1Score()

	# @brief Since we don't care about a particular threshold, just to get the highest activation for each prediction,
	#  we can compute the max on the last axis (classes axis) and send this as threshold to the ThresholdAccuracy
	#  class.
	def __call__(self, results : np.ndarray, labels : np.ndarray, **kwargs) -> float: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		return self.thresholdF1Score(results, labels, Max)
