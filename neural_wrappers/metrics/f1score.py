import numpy as np
from .metric import Metric
from .precision import Precision
from .recall import Recall

from ..utilities import npGetInfo

f1ScoreObj = None

class GlobalScores(Metric):
	def __init__(self):
		super().__init__()
		self.reset()

	def fReturn(self):
		if self.TP is None or self.FP is None or self.FN is None or self.TN is None:
			return 0, 0, 0, 0
		return self.TP, self.FP, self.FN, self.TN

	def reset(self):
		self.TP = None
		self.FP = None
		self.FN = None
		self.TN = None

	def onEpochStart(self, **kwargs):
		self.reset()

	def epochReduceFunction(self, results):
		res = self.fReturn()
		self.reset()
		return res

	def __call__(self, y, t, **k):
		NC = y.shape[-1]
		Max = y.max(axis=-1, keepdims=True)
		y = y >= Max
		t = t.astype(np.bool)

		if self.TP is None:
			self.TP = np.zeros((NC, ), dtype=np.int64)
			self.FP = np.zeros((NC, ), dtype=np.int64)
			self.FN = np.zeros((NC, ), dtype=np.int64)
			self.TN = np.zeros((NC, ), dtype=np.int64)
		# Sanity check to ensure we get the same amount of classes during iterations.
		assert len(self.TP.shape) == 1 and self.TP.shape[0] == NC

		for i in range(NC):
			tClass = t[..., i]
			yClass = y[..., i]
			TPClass = (yClass * tClass).sum()
			FPClass = (yClass * (1 - tClass)).sum()
			FNClass = ((1 - yClass) * tClass).sum()
			TNClass = ((1 - yClass) * (1 - tClass)).sum()

			self.TP[i] += TPClass
			self.FP[i] += FPClass
			self.FN[i] += FNClass
			self.TN[i] += TNClass
		return np.zeros((NC, ))

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
