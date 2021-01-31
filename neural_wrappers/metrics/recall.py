import numpy as np
from .metric import Metric
from ..utilities import NWNumber

recallObj = None

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-recall-and-recall-9250280bddc2
class Recall(Metric):
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

	def __call__(self, results:np.ndarray, labels:np.ndarray, **kwargs) -> NWNumber: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		results = np.uint8(results >= Max)
		# Nans are used to specify classes with no labels for this batch
		recall = Recall.computeRecall(results, labels)
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

def recall(y, t):
	global recallObj
	if not recallObj:
		recallObj = Recall()
	return recallObj(y, t)