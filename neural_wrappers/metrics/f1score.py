import numpy as np
from .global_scores import GlobalScores
from .metric import Metric
from .precision import Precision
from .recall import Recall

from ..utilities import npGetInfo

f1ScoreObj = None

class GlobalF1Score(Metric):
	def __init__(self):
		super().__init__(direction="max")
		self.globalScores = GlobalScores()

	def fReturn(self):
		TP, FP, FN, _ = self.globalScores.fReturn()
		precision = TP / (TP + FP + np.spacing(1))
		recall = TP / (TP + FN + np.spacing(1))
		f1 = 2 * precision * recall / (precision + recall + np.spacing(1))
		return f1

	def onEpochStart(self, **kwargs):
		self.globalScores.onEpochStart(**kwargs)

	def epochReduceFunction(self, results):
		res = self.fReturn()
		self.globalScores.epochReduceFunction(results)
		return res

	def iterationReduceFunction(self, results):
		return self.fReturn()

	def __call__(self, y, t, **k):
		self.globalScores.__call__(y, t, **k)
		return np.zeros((y.shape[-1], ))

class LocalF1Score(Metric):
	def __init__(self):
		super().__init__(direction="max")
		
	def __call__(self, results:np.ndarray, labels:np.ndarray, **kwargs) -> float: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		results = np.uint8(results >= Max)

		# Get Precision and Recall for this batch and apply the formula
		precision = Precision.computePrecision(results, labels)
		recall = Recall.computeRecall(results, labels)
		f1Score = 2 * precision * recall / (precision + recall + np.spacing(1))

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

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-f1Score-and-f1Score-9250280bddc2
class F1Score(Metric):
	def __init__(self, mode="local", returnMean:bool=True):
		super().__init__("max")
		assert mode in ("local", "global")
		self.returnMean = returnMean
		self.mode = mode
		self.obj = {
			"local" : LocalF1Score,
			"global" : GlobalF1Score
		}[mode]()

	def iterationReduceFunction(self, results):
		return self.obj.iterationReduceFunction(results).mean()

	def epochReduceFunction(self, results):
		results = self.obj.epochReduceFunction(results)
		if self.returnMean:
			results = results.mean()
		return results

	def __call__(self, y, t, **k):
		return self.obj(y, t, **k)

def f1score(y, t):
	global f1ScoreObj
	if f1ScoreObj is None:
		f1ScoreObj = F1Score(mode="local")
	return f1ScoreObj(y, t)
