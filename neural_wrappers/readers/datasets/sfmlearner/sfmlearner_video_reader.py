import pims
import numpy as np
from typing import Dict
from .sfmlearner_generic_reader import SfmLearnerGenericReader

class SfmLearnerVideoReader(SfmLearnerGenericReader):
	# @param[in] dataSplitMode Three options are available (for let's say 2 groups Train and Validation):
	#  - Random: Any index can be either T or V with probability given by dataSplits (T(1)T(2)V(3)T(4)V(5)T(6)..V(N))
	#  - Random (no overlap): Any index can be either T or V, but if an sequence subindex is in T(so [T-k,..T+k], given
	#     a sequenceSize of 2k), then V cannot have it's subindexes in that interval (so V+k<T-k or V-k>T+k)
	#  - Sequential: Ordered by the order of data split mode and no randomness: T(t1)T(t2)..T(tN)V(v1)...V(vN)
	#  - Sequential then random: Ordered by the order of data split, but inner order is random:
	#     T(ti1)T(ti2)T(ti3)..T(tiN)V(vi1)V(vi2)...V(viN) where ti1..tiN, vi1..viN is a randomized order

	def __init__(self, videoPath : str, sequenceSize : int, intrinsics : np.ndarray, \
		dataSplits : Dict[str, float]={"train" : 1}, dataSplitMode="random"):
		assert sequenceSize > 1
		assert dataSplitMode in ("random", "sequential", "sequential_then_random", "random_no_overlap")
		assert sum(dataSplits.values()) == 1
		self.videoPath = videoPath
		self.video = pims.Video(self.videoPath)
		self.intrinsics = intrinsics
		self.sequenceSize = sequenceSize
		self.dataSplits = dataSplits
		self.dataSplitMode = dataSplitMode
		
		self.dataSplitIndices = self.computeIndices()

	def getStartAndEndIndex(self):
		nTotal = len(self.video)
		# [0, 1, .., 9]. seqSize=2 => [0:8], seqSize=3 => [1:8], seqSize=4 => [1:7], seqSize=5 => [2:7] etc.
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
		return {k : permutation[range(*indices[k])] for k in self.dataSplits}

	def computeIndices(self):
		assert self.sequenceSize < len(self.video) - 2, "Sequence size: %d. Len video: %d" % \
			(self.sequenceSize, len(video))
		assert self.dataSplitMode == "random"
		np.random.seed(42)
		return {
			"random" : self.computeIndicesRandom
		}[self.dataSplitMode]()

	def getNumData(self, topLevel : str) -> int:
		return len(self.dataSplitIndices[topLevel])

	def __str__(self) -> str:
		Str = "[SfmLearnerVideoReader]"
		Str += "\n - Path: %s" % (self.videoPath)
		Str += "\n - Num frames: %d. FPS: %2.3f. Frame shape: %s" % \
			(len(self.video), self.video.frame_rate, self.video.frame_shape)
		Str += "\n - Sequence size: %d" % (self.sequenceSize)
		Str += "\n - Intrinsics: %s" % (self.intrinsics.tolist())
		Str += "\n - Data splits: %s" % (self.dataSplits)
		Str += "\n - Data split counts: %s" % ({k : len(self.dataSplitIndices[k]) for k in self.dataSplits})
		Str += "\n - Data split mode: %s" % (self.dataSplitMode)
		return Str