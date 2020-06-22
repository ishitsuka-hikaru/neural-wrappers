import pims
import numpy as np
from functools import partial
from typing import Dict, Any
from .sfmlearner_generic_reader import SfmLearnerGenericReader
from ...internal import DatasetRandomIndex, DatasetIndex
from ....utilities import smartIndexWrapper, npGetInfo

def defaultRgbGetter(dataset, index, sequenceSize):
	l, r = sequenceSize // 2 - (sequenceSize % 2 == 0), sequenceSize // 2
	extendedIndices = [list(range(x - l, x + r + 1)) for x in index.sequence]
	items = smartIndexWrapper(dataset, extendedIndices)
	return items

# Since we know that the index is sequential, there is no need to go the default way, and instead we can read all the
#  items at once sequentially and then create a smart index in the returned contiguous array
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] => [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
#  [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12]]
def sequentialRgbGetter(dataset, index, sequenceSize):
	l, r = sequenceSize // 2 - (sequenceSize % 2 == 0), sequenceSize // 2
	extendedIndices = np.array([list(range(x - l, x + r + 1)) for x in index.sequence])
	fastIndex = list(range(index.sequence[0] - l, index.sequence[-1] + r + 1))
	extendedFastIndices = extendedIndices - index.sequence[0] + l
	fastItems = np.array(dataset[fastIndex])
	fastItems = fastItems[extendedFastIndices]

	return fastItems

def rgbNorm(x):
	return ((np.float32(x) / 255) - 0.5) * 2

class SfmLearnerVideoReader(SfmLearnerGenericReader):
	# @param[in] dataSplitMode Three options are available (for let's say 2 groups Train and Validation):
	#  - Random: Any index can be either T or V with probability given by dataSplits (T(1)T(2)V(3)T(4)V(5)T(6)..V(N))
	#  - Random (no overlap): Any index can be either T or V, but if an sequence subindex is in T(so [T-k,..T+k], given
	#     a sequenceSize of 2k), then V cannot have it's subindexes in that interval (so V+k<T-k or V-k>T+k)
	#  - Sequential: Ordered by the order of data split mode and no randomness: T(t1)T(t2)..T(tN)V(v1)...V(vN)
	#  - Sequential then random: Ordered by the order of data split, but inner order is random:
	#     T(ti1)T(ti2)T(ti3)..T(tiN)V(vi1)V(vi2)...V(viN) where ti1..tiN, vi1..viN is a randomized order

	def __init__(self, videoPath : str, sequenceSize : int, intrinsics : np.ndarray, \
		dataSplits : Dict[str, float]={"train" : 1}, dataSplitMode="random", videoMode="fast"):
		assert sequenceSize > 1
		assert dataSplitMode in ("random", "sequential", "sequential_then_random", "random_no_overlap")
		assert sum(dataSplits.values()) == 1
		self.videoPath = videoPath
		self.video = pims.Video(self.videoPath)
		self.fps = self.video.frame_rate
		if videoMode == "fast":
			self.video = self.video.close()
			self.video = pims.PyAVReaderIndexed(self.videoPath)

		self.intrinsics = intrinsics
		self.sequenceSize = sequenceSize
		self.dataSplits = dataSplits
		self.dataSplitMode = dataSplitMode
		
		self.dataSplitIndices = self.computeIndices()
		rgbGetter = {
			"random" : defaultRgbGetter,
			"sequential" : sequentialRgbGetter
		}[self.dataSplitMode]
		super().__init__(dataBuckets={"data" : ["rgb", "intrinsics"]}, \
			dimGetter={"rgb" : partial(rgbGetter, sequenceSize=self.sequenceSize), \
				"intrinsics" : (lambda dataset, index : self.intrinsics)}, \
			dimTransform={"data" : {"rgb" : rgbNorm}}
		)

	def getStartAndEndIndex(self):
		nTotal = len(self.video)
		# [0, 1, .., 9]. sequenceSize=2 => [0:8], sequenceSize=3 => [1:8], sequenceSize=4 => [1:7],
		#  sequenceSize=5 => [2:7] etc.
		startIndex = self.sequenceSize // 2 - 1 + (self.sequenceSize % 2 == 1)
		endIndex = nTotal - (self.sequenceSize // 2) - 1
		return startIndex, endIndex

	def computeIndicesRandom(self):
		startIndex, endIndex = self.getStartAndEndIndex()
		n = endIndex - startIndex
		permutation = np.random.permutation(n)

		# Now, the permutation is for all the dataset. We need to chop it properly.
		indices = {}
		currentStart = 0
		for k in self.dataSplits:
			nCurrent = int(self.dataSplits[k] * n)
			indices[k] = (currentStart, currentStart + nCurrent)
			currentStart += nCurrent
		# Last key has the remaining float-to-int error frames as well
		indices[k] = (indices[k][0], n)
		indices = {k : range(indices[k][0], indices[k][1]) for k in indices}
		indices = {k : startIndex + permutation[indices[k]] for k in indices}
		return indices

	def computeIndicesSequential(self):
		startIndex, endIndex = self.getStartAndEndIndex()
		n = endIndex - startIndex

		indices = {}
		currentStart = startIndex
		for k in self.dataSplits:
			nCurrent = int(self.dataSplits[k] * n)
			indices[k] = (currentStart, currentStart + nCurrent)
			currentStart += nCurrent
		indices[k] = (indices[k][0], n)
		indices = {k : range(indices[k][0], indices[k][1]) for k in indices}
		return indices

	def computeIndices(self):
		assert self.sequenceSize < len(self.video) - 2, "Sequence size: %d. Len video: %d" % \
			(self.sequenceSize, len(video))
		np.random.seed(42)
		return {
			"random" : self.computeIndicesRandom,
			"sequential" : self.computeIndicesSequential
		}[self.dataSplitMode]()

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
		Str += "\n - Num frames: %d. FPS: %2.3f. Frame shape: %s" % \
			(len(self.video), self.fps, self.video.frame_shape)
		Str += "\n - Sequence size: %d" % (self.sequenceSize)
		Str += "\n - Intrinsics: %s" % (self.intrinsics.tolist())
		Str += "\n - Data splits: %s" % (self.dataSplits)
		Str += "\n - Data split counts: %s" % ({k : len(self.dataSplitIndices[k]) for k in self.dataSplits})
		Str += "\n - Data split mode: %s" % (self.dataSplitMode)
		return Str