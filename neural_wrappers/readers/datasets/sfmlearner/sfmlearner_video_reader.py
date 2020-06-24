import pims
import numpy as np
from functools import partial
from typing import Dict, Any, Callable
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

def intrinsicGetter(dataset, index, fieldOfView, nativeResolution, desiredResolution):
	cy = nativeResolution[0] / 2
	cx = nativeResolution[1] / 2
	fy = cy / (np.tan(fieldOfView * np.pi / 360))
	fx = cx / (np.tan(fieldOfView * np.pi / 360))
	skew = 0

	sy = desiredResolution[0] / nativeResolution[0]
	sx = desiredResolution[1] / nativeResolution[1]
	K = np.array([
		[fx * sx, skew, cx * sx],
		[0, fy * sy, cy * sy],
		[0, 0, 1]
	], dtype=np.float32)
	return K

class SfmLearnerVideoReader(SfmLearnerGenericReader):
	# @param[in] videoPath Path to the video file
	# @param[in] sequenceSize The length of the sequence (nearby frames) returned at each iteration
	# @param[in] cameraParams A dictionary wtih field of fiew and native resolution keys
	# @param[in] dataSplitMode Three options are available (for let's say 2 groups Train and Validation):
	#  - Random: Any index can be either T or V with probability given by dataSplits (T(1)T(2)V(3)T(4)V(5)T(6)..V(N))
	#  - Random (no overlap): Any index can be either T or V, but if an sequence subindex is in T(so [T-k,..T+k], given
	#     a sequenceSize of 2k), then V cannot have it's subindexes in that interval (so V+k<T-k or V-k>T+k)
	#  - Sequential: Ordered by the order of data split mode and no randomness: T(t1)T(t2)..T(tN)V(v1)...V(vN)
	#  - Sequential then random: Ordered by the order of data split, but inner order is random:
	#     T(ti1)T(ti2)T(ti3)..T(tiN)V(vi1)V(vi2)...V(viN) where ti1..tiN, vi1..viN is a randomized order
	# @param[in] skipFrames How many frames ahead should be in each sequence. Default: 1.
	#  Example: skipFrames=2 and sequenceSize=5 => [t-4, t-2, t, t+2, t+4]

	def __init__(self, videoPath:str, sequenceSize:int, cameraParams:np.ndarray, dataSplits:Dict[str, float], \
		skipFrames:int = 1, dataSplitMode:str = "random", videoMode:str = "fast", \
		dimTransform:Dict[str, Dict[str, Callable]]={}):
		assert sequenceSize > 1
		assert sum(dataSplits.values()) == 1
		self.videoPath = videoPath
		self.video = pims.Video(self.videoPath)
		self.fps = self.video.frame_rate
		self.frameShape = self.video.frame_shape
		if videoMode == "fast":
			self.video = self.video.close()
			self.video = pims.PyAVReaderIndexed(self.videoPath)

		self.fieldOfView = cameraParams["fieldOfView"]
		self.nativeResolution = cameraParams["nativeResolution"]
		self.sequenceSize = sequenceSize
		self.skipFrames = skipFrames
		self.dataSplits = dataSplits
		self.dataSplitMode = dataSplitMode
		self.dataSplitIndices = computeIndices(self.dataSplitMode, self.dataSplits, len(self.video), \
			self.sequenceSize, self.skipFrames)

		rgbGetter = {
			"random" : defaultRgbGetter,
			"sequential" : sequentialRgbGetter,
			"sequential_then_random" : defaultRgbGetter
		}[self.dataSplitMode]
		fIntrinsicsGetter = partial(intrinsicGetter, fieldOfView=self.fieldOfView, \
			nativeResolution=self.nativeResolution, desiredResolution=self.frameShape)

		super().__init__(
			dataBuckets={"data" : ["rgb", "intrinsics"]}, \
			dimGetter={"rgb" : rgbGetter, "intrinsics" : fIntrinsicsGetter}, \
			dimTransform=dimTransform
		)

	def getNumData(self, topLevel : str) -> int:
		return len(self.dataSplitIndices[topLevel])

	def getDataset(self, topLevel : str) -> Any:
		return self.video

	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		startIndex = i * batchSize
		endIndex = min((i + 1) * batchSize, self.getNumData(topLevel))
		indices = self.dataSplitIndices[topLevel][startIndex : endIndex]
		return DatasetRandomIndex(indices)

	def __str__(self) -> str:
		Str = "[SfmLearnerVideoReader]"
		Str += "\n - Path: %s" % (self.videoPath)
		Str += "\n - Num frames: %d. FPS: %2.3f. Frame shape: %s" % (len(self.video), self.fps, self.frameShape)
		Str += "\n - Sequence size: %d. Skip frames: %d" % (self.sequenceSize, self.skipFrames)
		Str += "\n - FoV: %d. Native resolution: %s" % (self.fieldOfView, self.nativeResolution)
		K = intrinsicGetter(None, None, self.fieldOfView, self.nativeResolution, self.frameShape)
		Str += "\n - Intrinsic camera: %s" % (K.tolist())
		Str += "\n - Data splits: %s" % (self.dataSplits)
		Str += "\n - Data split counts: %s" % ({k : len(self.dataSplitIndices[k]) for k in self.dataSplits})
		Str += "\n - Data split mode: %s" % (self.dataSplitMode)
		return Str