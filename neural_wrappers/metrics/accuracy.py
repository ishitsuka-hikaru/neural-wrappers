import numpy as np
from overrides import overrides
from typing import Dict
from .metric import Metric
from ..utilities import NWNumber

accObj = None

class Accuracy(Metric):
	def __init__(self, meanResult:bool=True):
		super().__init__(direction="max")
		self.meanResult = meanResult

	@overrides
	def getExtremes(self) -> Dict[str, NWNumber]:
		return {"min" : 0, "max" : 1}

	def __call__(self, results:np.ndarray, labels:np.ndarray, **kwargs) -> float: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		binaryResults = results >= Max
		whereCorrect = labels == 1
		
		maskedResults = binaryResults[whereCorrect]
		maskedResults = maskedResults.reshape(*labels.shape[0 : -1])
		if self.meanResult:
			maskedResults = maskedResults.mean()
		return maskedResults

# Simple wrapper for the Accuracy class
# @param[in] y Predictions (After softmax). Shape: MBx(Shape)xNC
# @param[in] t Class labels. Shape: MBx(Shape) and values of 0 and 1.
def accuracy(y:np.ndarray, t:np.ndarray):
	global accObj
	if accObj is None:
		accObj = Accuracy(meanResult=False)
	return accObj(y, t)