import numpy as np
from typing import Dict

# Utility functions
def getVideoExtremes(videoLength:int, sequenceSize:int, skipFrames:int):
	# left: avoid these indices that would get me to negative counts for the biggest first index given a sequence
	#  size and skipFrames. So S=5, skipFrames=2 means that for the first item, the center needs to be in [4].
	#  Becuase the first item has indexes: [0, 2, 4, 6, 8]
	l = ((sequenceSize // 2 - (sequenceSize % 2 == 0))) * skipFrames
	r = videoLength - ((sequenceSize // 2) * skipFrames) - 1
	return l, r

# Indices functions
def computeIndicesRandom(indices:np.ndarray, sequenceSize:int, skipFrames:int):
	seqIndices = computeIndicesSequential(indices, sequenceSize, skipFrames)
	np.random.seed(42)
	perm = np.random.permutation(len(seqIndices))
	return seqIndices[perm]

def computeIndicesSequential(indices:np.ndarray, sequenceSize:int, skipFrames:int):
	l, r = sequenceSize // 2 - (sequenceSize % 2 == 0), sequenceSize // 2
	leftRightRange = np.arange(-l * skipFrames, (r * skipFrames) + 1, skipFrames)
	seqIndices = np.array([leftRightRange + x for x in indices], dtype=np.int32)
	return seqIndices

# Split functions
def getRandomSplits(videoLength:int, dataSplits:Dict[str, float], sequenceSize:int, skipFrames:int):
	sequentialSplits = getSequentialSplits(videoLength, dataSplits, sequenceSize, skipFrames)

	randomSplits = {}
	np.random.seed(42)
	for k in sequentialSplits:
		perm = np.random.permutation(len(sequentialSplits[k]))
		randomSplits[k] = sequentialSplits[k][perm]
	return randomSplits

def getSequentialSplits(videoLength:int, dataSplits:Dict[str, float], sequenceSize:int, skipFrames:int):
	l, r = getVideoExtremes(videoLength, sequenceSize, skipFrames)
	validVideoLength = r - l
	startIndex = l

	indices = {}
	for k in dataSplits:
		endIndex = startIndex + int(dataSplits[k] * validVideoLength)
		indices[k] = (startIndex, endIndex)
		startIndex = endIndex
	# Last split gets all the extra indexes as well
	indices[k] = (indices[k][0], r)
	return {k : np.array(range(indices[k][0], indices[k][1]), dtype=np.int32) for k in dataSplits}

def computeIndices(dataSplitMode:str, dataSplits:Dict[str, float], videoLength:int, sequenceSize:int, skipFrames:int):
	assert dataSplitMode in ("random", "sequential", "sequential_then_random", "random_no_overlap")
	assert sequenceSize < videoLength - 2, "Sequence size: %d. Len video: %d" % (sequenceSize, videoLength)

	# This is a two process step. First, we need to divide the data given the data splits. This takes the pressure
	#  off of the algorithms to do this step as well as compute their logic. The result is the middle frame of the
	#  video, irregardless of the sequence Size.
	# Example: videoLength=10, and dataSplits={"train": 0.8, "validation": 0.2}, we'll get 8 indices for "train"
	#  and 2 indices for "validation".
	splitFn = {
		"random" : getRandomSplits,
		"sequential" : getSequentialSplits,
		"sequential_then_random" : getSequentialSplits
	}[dataSplitMode]
	dataSplitIndexes = splitFn(videoLength, dataSplits, sequenceSize, skipFrames)

	indicesFn = {
		"random" : computeIndicesRandom,
		"sequential" : computeIndicesSequential,
		"sequential_then_random" : computeIndicesRandom
	}[dataSplitMode]
	result = {}
	for split in dataSplitIndexes:
		result[split] = indicesFn(dataSplitIndexes[split], sequenceSize, skipFrames)
		# Since the sequence sizes could mean we have indices outside the possible range if badly computed.
		assert np.logical_or(result[split] < 0, result[split] >= videoLength).sum() == 0
	return result