import numpy as np
from functools import partial
from typing import Dict, Any, Callable, List
from overrides import overrides

from .sfmlearner_generic_reader import SfmLearnerGenericReader
from ...internal import DatasetRandomIndex, DatasetIndex
from ....utilities import smartIndexWrapper, npGetInfo

def defaultRgbGetter(dataset, index):
	items = smartIndexWrapper(dataset, index.sequence)
	return items

class SfmLearnerVideoReader(SfmLearnerGenericReader):
	# @param[in] datasetPath Path to the video file
	# @param[in] sequenceSize The length of the sequence (nearby frames) returned at each iteration
	# @param[in] intrinsicMatrix The intrinsic matrix used by the camera to record the video. Defaults to np.eye(3)
	# @param[in] dataSplitMode Three options are available (for let's say 2 groups Train and Validation):
	#  - Random: Any index can be either T or V with probability given by dataSplits (T(1)T(2)V(3)T(4)V(5)T(6)..V(N))
	#  - Random (no overlap): Any index can be either T or V, but if an sequence subindex is in T(so [T-k,..T+k], given
	#     a sequenceSize of 2k), then V cannot have it's subindexes in that interval (so V+k<T-k or V-k>T+k)
	#  - Sequential: Ordered by the order of data split mode and no randomness: T(t1)T(t2)..T(tN)V(v1)...V(vN)
	#  - Sequential then random: Ordered by the order of data split, but inner order is random:
	#     T(ti1)T(ti2)T(ti3)..T(tiN)V(vi1)V(vi2)...V(viN) where ti1..tiN, vi1..viN is a randomized order
	# @param[in] skipFrames How many frames ahead should be in each sequence. Default: 1.
	#  Example: skipFrames=2 and sequenceSize=5 => [t-4, t-2, t, t+2, t+4]

	def __init__(self, datasetPath:str, sequenceSize:int, dataSplitIndices:Dict[str, List[int]], \
		intrinsicMatrix:np.ndarray = np.eye(3), dimTransform:Dict[str, Dict[str, Callable]]={}):
		import pims
		self.datasetPath = datasetPath
		self.video = pims.Video(self.datasetPath)
		self.fps = self.video.frame_rate
		self.frameShape = self.video.frame_shape
		self.dataSplitIndices = dataSplitIndices

		super().__init__(
			dataBuckets = {"data" : ["rgb", "intrinsics"]}, \
			dimGetter = {"rgb" : defaultRgbGetter, "intrinsics" : (lambda _, __ : intrinsicMatrix)}, \
			dimTransform = dimTransform,
			dataSplitIndices = dataSplitIndices,
			intrinsicMatrix = intrinsicMatrix
		)

	@overrides
	def getNumData(self, topLevel : str) -> int:
		return len(self.dataSplitIndices[topLevel])

	@overrides
	def getDataset(self, topLevel : str) -> Any:
		return self.video

	@overrides
	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		startIndex = i * batchSize
		endIndex = min((i + 1) * batchSize, self.getNumData(topLevel))
		indices = self.dataSplitIndices[topLevel][startIndex : endIndex]
		return DatasetRandomIndex(indices)

	def __str__(self) -> str:
		Str = "[SfmLearnerVideoReader]"
		Str += "\n - Path: %s" % (self.datasetPath)
		Str += "\n - Resolution: %d x %d" % (self.frameShape[0], self.frameShape[1])
		Str += "\n - Num frames: %d. FPS: %2.3f" % (len(self.video), self.fps)
		Str += "\n - Sequence size: %d. Skip frames: %d." % (self.sequenceSize, self.skipFrames)
		Str += "\n - Intrinsic camera: %s" % (self.intrinsicMatrix.tolist())
		Str += "\n - Data splits: %s" % (list(self.dataSplitIndices.keys()))
		Str += "\n - Data split counts: %s" % ({k : len(self.dataSplitIndices[k]) for k in self.dataSplitIndices})
		return Str

