import numpy as np
from .metric import Metric

precisionObj = None

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
class Precision(Metric):
	def __init__(self):
		super().__init__("max")

	def computePrecision(results:np.ndarray, labels:np.ndarray) -> np.ndarray:
		TP = results * labels
		FP = results * (1 - labels)
		TP = TP.sum(axis=0)
		FP = FP.sum(axis=0)
		res = (TP / (TP + FP + 1e-8))

		# We only care about those results that have actual labels (so we don't get 0 precision for a class that has
		#  no valid prediction in this MB). We mask the irrelevant classes with nans.
		whereOk = (labels.sum(axis=0) > 0).astype(np.float32)
		whereOk[whereOk == 0] = np.nan

		res = res * whereOk
		return res

	def __call__(self, results:np.ndarray, labels:np.ndarray, **kwargs) -> float: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		results = np.uint8(results >= Max)
		# Nans are used to specify classes with no labels for this batch
		precision = Precision.computePrecision(results, labels)
		# Keep only position where precision is not nan.
		whereNotNaN = ~np.isnan(precision)
		precision = precision[whereNotNaN]
		# Mean over those valid classes.
		# return precision.mean()

		# It's better to compute the weighted mean of these predictions, instead of treating each element in this
		#  MB equally.
		whereOk = labels.sum(axis=0)
		whereOk = whereOk[whereNotNaN]
		return (precision * whereOk).sum() / whereOk.sum()

def precision(y, t):
	global precisionObj
	if not precisionObj:
		precisionObj = Precision()
	return precisionObj(y, t)