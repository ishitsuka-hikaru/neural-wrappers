import numpy as np
from overrides import overrides
from typing import Dict
from .metric import Metric
from ..utilities import NWNumber

accObj = {}

class Accuracy(Metric):
	def __init__(self, returnMean:bool=True):
		super().__init__(direction="max")
		self.returnMean = returnMean

	@overrides
	def getExtremes(self) -> Dict[str, NWNumber]:
		return {"min" : 0, "max" : 1}

	def __call__(self, results:np.ndarray, labels:np.ndarray, **kwargs) -> float: #type: ignore[override]	
		assert len(np.unique(labels)) == 2
		Shape = labels.shape[0 : -1]
		labels = labels.astype(bool)
		binaryResults = results == results.max(axis=-1, keepdims=True)
		maskedResults = binaryResults[labels].reshape(*Shape)
		if self.returnMean:
			maskedResults = maskedResults.mean()
		return maskedResults

# Simple wrapper for the Accuracy class
# @param[in] y Predictions (After softmax). Shape: MBx(Shape)xNC
# @param[in] t Class labels. Shape: MBx(Shape)xNC and values of 0 and 1.
def accuracy(y:np.ndarray, t:np.ndarray, returnMean:bool=False):
	global accObj
	if not returnMean in accObj:
		accObj[returnMean] = Accuracy(returnMean=returnMean)
	return accObj[returnMean](y, t)