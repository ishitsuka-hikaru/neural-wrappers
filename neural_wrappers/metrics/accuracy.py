import numpy as np
from overrides import overrides
from typing import Dict
from .metric import Metric
from ..utilities import NWNumber
from .global_scores import GlobalScores

accObj = None

class GlobalAccuracy(Metric):
	def __init__(self):
		super().__init__(direction="max")
		self.reset()

	def fReturn(self):
		return self.globalCorrect / (self.globalAll + np.spacing(1))

	def reset(self):
		self.globalCorrect = 0
		self.globalAll = 0

	def onEpochStart(self, **kwargs):
		self.reset()

	def epochReduceFunction(self, results):
		res = self.fReturn()
		self.reset()
		return res

	def __call__(self, y, t, **k):
		t = t.astype(bool)
		y = (y == y.max(axis=-1, keepdims=True))
		y = y[t]
		self.globalCorrect += y.sum()
		self.globalAll += t.sum()
		return self.fReturn()

class LocalAccuracy(Metric):
	def __init__(self):
		super().__init__(direction="max")

	def __call__(self, results:np.ndarray, labels:np.ndarray, **kwargs) -> float: #type: ignore[override]	
		assert len(np.unique(labels)) == 2
		labels = labels.astype(bool)
		binaryResults = results == results.max(axis=-1, keepdims=True)
		maskedResults = binaryResults[labels]
		return maskedResults.sum() / labels.sum()

# Based on https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-f1Score-and-f1Score-9250280bddc2
class Accuracy(Metric):
	def __init__(self, mode="local"):
		super().__init__("max")
		assert mode in ("local", "global")
		self.mode = mode
		self.obj = {
			"local" : LocalAccuracy,
			"global" : GlobalAccuracy
		}[mode]()

	@overrides
	def getExtremes(self) -> Dict[str, NWNumber]:
		return {"min" : 0, "max" : 1}

	def iterationReduceFunction(self, results):
		return self.obj.iterationReduceFunction(results)

	def epochReduceFunction(self, results):
		results = self.obj.epochReduceFunction(results)
		return results

	def __call__(self, y, t, **k):
		return self.obj(y, t, **k)

# Simple wrapper for the Accuracy class
# @param[in] y Predictions (After softmax). Shape: MBx(Shape)xNC
# @param[in] t Class labels. Shape: MBx(Shape) and values of 0 and 1.
def accuracy(y:np.ndarray, t:np.ndarray):
	global accObj
	if accObj is None:
		accObj = Accuracy(mode="local")
	return accObj(y, t)
