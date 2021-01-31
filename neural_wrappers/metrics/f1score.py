import numpy as np
from .metric import Metric
from .precision import Precision
from .recall import Recall

from ..utilities import npGetInfo

f1ScoreObj = None

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-f1Score-and-f1Score-9250280bddc2
class F1Score(Metric):
	def __init__(self):
		super().__init__("max")

	def computeF1Score(results: np.ndarray, labels : np.ndarray) -> np.ndarray:
		precision = Precision.computePrecision(results, labels)
		recall = Recall.computeRecall(results, labels)
		f1Score = 2 * precision * recall / (precision + recall + np.spacing(1))
		return f1Score
		
	def __call__(self, results : np.ndarray, labels : np.ndarray, **kwargs) -> float: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		results = np.uint8(results >= Max)
		# Nans are used to specify classes with no labels for this batch
		f1Score = F1Score.computeF1Score(results, labels)
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

def f1score(y, t):
	global f1ScoreObj
	if f1ScoreObj is None:
		f1ScoreObj = F1Score()
	return f1ScoreObj(y, t)