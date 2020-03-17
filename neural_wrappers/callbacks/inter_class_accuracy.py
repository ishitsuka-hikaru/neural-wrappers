import numpy as np
from .metric_as_callback import MetricAsCallback
from ..utilities import Number

class InterClassAccuracy(MetricAsCallback):
	def __init__(self):
		super().__init__("InterClassAccuracy")

	def getDirection(self):
		return "max"

	def epochReduceFunction(self, results) -> Number:
		return results.mean()

	def iterationReduceFunction(self, results):
		return results

	def onIterationEnd(self, results, labels, **kwargs):
		Max = results.max(axis=-1, keepdims=True)
		results = np.uint8(results >= Max)
		XOR = 1 - np.logical_xor(labels, results)
		# Mean just the batch, so we have a mean PER class
		XOR = XOR.mean(axis=0)
		return XOR