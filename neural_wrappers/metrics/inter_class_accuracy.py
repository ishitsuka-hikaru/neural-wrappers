import numpy as np
from overrides import overrides
from .metric import Metric
from ..utilities import NWNumber

accObj = None

class InterClassAccuracy(Metric):
	def __init__(self):
		super().__init__(direction="max")
		self.setName("InterClassAccuracy")

	@overrides
	def epochReduceFunction(self, results:np.ndarray) -> NWNumber:
		return results.mean()

	@overrides
	def iterationReduceFunction(self, results:np.ndarray) -> NWNumber:
		return results

	@overrides
	def onIterationEnd(self, results:np.ndarray, labels:np.ndarray, **kwargs) -> NWNumber:
		Max = results.max(axis=-1, keepdims=True)
		results = np.uint8(results >= Max)
		XOR = 1 - np.logical_xor(labels, results)
		# Mean just the batch, so we have a mean PER class
		XOR = XOR.mean(axis=0)
		return XOR

def inter_class_accuracy(y, t):
	global accObj
	if accObj is None:
		accObj = InterClassAccuracy()
	return accObj(y, t)