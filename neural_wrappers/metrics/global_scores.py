import numpy as np
from .metric import Metric

# @brief Helper class to compute global counters over an entire dataset. At the end of the epoch, the global counters
#  are reset and the values are returned so we can have a non batch-averaged result.
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