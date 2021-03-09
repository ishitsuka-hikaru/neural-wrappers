import numpy as np
from scipy.special import softmax
from .metric import Metric

meanIoUObj = {}

class GlobalF1Score(Metric):
	def __init__(self):
		super().__init__(direction="max")
		self.globalScores = GlobalScores()

	def fReturn(self):
		TP, FP, FN, _ = self.globalScores.fReturn()
		IoU = TP / (TP + FP + FN + np.spacing(1))
		return IoU

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

# Compute the global mean iou, which is updated on a per epoch mode.
class GlobalMeanIoU(Metric):
	def __init__(self):
		super().__init__(direction="max")
		self.unions = None
		self.intersections = None

	def fReturn(self):
		if self.unions is None:
			return 0
		whereNotZero = self.unions != 0
		res = (self.intersections[whereNotZero] / self.unions[whereNotZero])
		return res

	def epochReduceFunction(self, results):
		res = self.fReturn()
		self.intersections *= 0
		self.unions *= 0
		return res

	def iterationReduceFunction(self, results):
		return (self.intersections / (self.unions + 1e-5))

	def __call__(self, y, t, **k):
		Max = y.max(axis=-1, keepdims=True)
		y = y >= Max
		t = t.astype(np.bool)

		NC = y.shape[-1]
		if self.intersections is None:
			self.intersections = np.zeros((NC, ), dtype=np.uint64)
			self.unions = np.zeros((NC, ), dtype=np.uint64)
		assert len(self.unions.shape) == 1 and self.unions.shape[0] == NC

		for i in range(NC):
			tClass = t[..., i]
			yClass = y[..., i]

			intersectionClass = (yClass * tClass).sum()
			unionClass = ((yClass + tClass) > 0).sum()

			self.intersections[i] += intersectionClass
			self.unions[i] += unionClass
		return np.array([0])

# Compute the batch (local) mean iou, which is an approximation of the actual mean iou over entire dataset, however
#  it is faster to compute. This allows us to use the running mean of the network class.
class LocalMeanIoU(Metric):
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
		return intersection.sum(axis=0) / (union.sum(axis=0) + 1e-5)

class MeanIoU(Metric):
	def __init__(self, mode="local", returnMean=True):
		super().__init__(direction="max")
		assert mode in ("local", "global")
		self.obj = {
			"local" : LocalMeanIoU,
			"global" : GlobalMeanIoU
		}[mode]()
		self.returnMean = returnMean

	def iterationReduceFunction(self, results):
		return self.obj.iterationReduceFunction(results).mean()

	def epochReduceFunction(self, results):
		results = self.obj.epochReduceFunction(results)
		if self.returnMean:
			results = results.mean()
		return results

	def __call__(self, y, t, **k):
		return self.obj(y, t, **k)

def mean_iou(y, t, returnMean:bool=False):
	global meanIoUObj
	singletonKey = (returnMean, )
	# We can only use local mIoU using this. For global mIoU, use the Callback itself.
	if not singletonKey in meanIoUObj:
		meanIoUObj[singletonKey] = MeanIoU(mode="local", returnMean=returnMean)
	return meanIoUObj[singletonKey](y, t)