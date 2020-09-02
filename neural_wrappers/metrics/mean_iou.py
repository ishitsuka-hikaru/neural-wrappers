import numpy as np
from scipy.special import softmax
from .metric import Metric

class MeanIoU(Metric):
	def __init__(self):
		super().__init__(direction="max")

	# @brief results and labels are two arrays of shape: (MB, N, C), where N is a variable shape (images, or just
	#  numbers), while C is the number of classes. The number of classes must coincide to both cases. We assume that
	#  the labels are one-hot encoded (1 on correct class, 0 on others), while we make no assumption about the results.
	# Thus: intersection is if both label and result are 1 while reunion is if either result or label are 1
	def __call__(self, results : np.ndarray, labels : np.ndarray, **kwargs) -> float: #type: ignore[override]
		Max = results.max(axis=-1, keepdims=True)
		results = results >= Max
		
		intersection = results * labels
		union = (results + labels) > 0
		return intersection.sum() / (union.sum() + 1e-5)