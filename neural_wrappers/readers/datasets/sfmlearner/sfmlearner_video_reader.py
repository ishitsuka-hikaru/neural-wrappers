import pims
import numpy as np
from functools import partial
from typing import Dict, Any, Callable
from overrides import overrides

from .sfmlearner_generic_reader import SfmLearnerGenericReader
from .video_utils import computeIndices
from ...internal import DatasetRandomIndex, DatasetIndex
from ....utilities import smartIndexWrapper, npGetInfo

def defaultRgbGetter(dataset, index):
	items = smartIndexWrapper(dataset, index.sequence)
	return items

# Since we know that the index is sequential, there is no need to go the default way, and instead we can read all the
#  items at once sequentially and then create a smart index in the returned contiguous array
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] => [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
#  [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12]]
def sequentialRgbGetter(dataset, index):
	Min, Max = index.sequence[0, 0], index.sequence[-1, -1] + 1
	indices = np.arange(Min, Max)
	fastItems = np.array(dataset[indices])
	sequenceIndices = index.sequence - Min
	items = fastItems[sequenceIndices]
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

	def __init__(self, datasetPath:str, sequenceSize:int, dataSplits:Dict[str, float], \
		intrinsicMatrix:np.ndarray = np.eye(3), skipFrames:int = 1, dataSplitMode:str = "random", \
		dimTransform:Dict[str, Dict[str, Callable]]={}):
		self.datasetPath = datasetPath
		self.video = pims.Video(self.datasetPath)
		self.fps = self.video.frame_rate
		self.frameShape = self.video.frame_shape

		self.skipFrames = skipFrames
		self.dataSplitMode = dataSplitMode
		self.dataSplitIndices = computeIndices(self.dataSplitMode, dataSplits, len(self.video), \
			sequenceSize, self.skipFrames)

		rgbGetter = {
			"random" : defaultRgbGetter,
			"sequential" : sequentialRgbGetter,
			"sequential_then_random" : defaultRgbGetter
		}[self.dataSplitMode]

		super().__init__(
			dataBuckets={"data" : ["rgb", "intrinsics"]}, \
			dimGetter={"rgb" : rgbGetter, "intrinsics" : (lambda _, __ : intrinsicMatrix)}, \
			dimTransform=dimTransform,
			sequenceSize=sequenceSize, dataSplits=dataSplits, intrinsicMatrix=intrinsicMatrix
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
		# Str += "\n - FoV: %d. Native resolution: %s" % (self.fieldOfView, self.nativeResolution)
		Str += "\n - Intrinsic camera: %s" % (self.intrinsicMatrix.tolist())
		Str += "\n - Data splits: %s" % (self.dataSplits)
		Str += "\n - Data split counts: %s" % ({k : len(self.dataSplitIndices[k]) for k in self.dataSplits})
		Str += "\n - Data split mode: %s" % (self.dataSplitMode)
		return Str

