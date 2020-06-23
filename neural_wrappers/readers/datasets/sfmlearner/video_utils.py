import numpy as np

# def computeIndicesRandom():
#     np.random.seed(42)
#     permutation = np.random.permutation(n)

#     # Now, the permutation is for all the dataset. We need to chop it properly.
#     indices = {}
#     currentStart = 0
#     for k in self.dataSplits:
#         nCurrent = int(self.dataSplits[k] * n)
#         indices[k] = (currentStart, currentStart + nCurrent)
#         currentStart += nCurrent
#     # Last key has the remaining float-to-int error frames as well
#     indices[k] = (indices[k][0], n)
#     indices = {k : range(indices[k][0], indices[k][1]) for k in indices}
#     indices = {k : startIndex + permutation[indices[k]] for k in indices}
#     return indices

# def computeIndicesSequential(videoLength, sequenceSize):
#     startIndex, endIndex = getStartAndEndIndex(videoLength, sequenceSize)
#     n = endIndex - startIndex

#     indices = {}
#     currentStart = startIndex
#     for k in self.dataSplits:
#         nCurrent = int(self.dataSplits[k] * n)
#         indices[k] = (currentStart, currentStart + nCurrent)
#         currentStart += nCurrent
#     indices[k] = (indices[k][0], n)
#     indices = {k : range(indices[k][0], indices[k][1]) for k in indices}
#     return indices

# Indices functions
def computeIndicesRandom(indices, sequenceSize):
	seqIndices = computeIndicesSequential(indices, sequenceSize)
	np.random.seed(42)
	perm = np.random.permutation(len(seqIndices))
	return seqIndices[perm]

def computeIndicesSequential(indices, sequenceSize):
	l, r = sequenceSize // 2 - (sequenceSize % 2 == 0), sequenceSize // 2 + 1
	seqIndices = np.array([list(range(x - l, x + r)) for x in indices], dtype=np.uint32)
	return seqIndices

# Split functions
def getRandomSplits(videoLength, dataSplits):
	np.random.seed(42)
	permutation = np.random.permutation(videoLength)
	sequentialSplits = getSequentialSplits(videoLength, dataSplits)
	randomSplits = {k : permutation[sequentialSplits[k]] for k in dataSplits}
	return randomSplits

def getSequentialSplits(videoLength, dataSplits):
	startIndex = 0
	indices = {}
	for k in dataSplits:
		endIndex = startIndex + int(dataSplits[k] * videoLength)
		indices[k] = (startIndex, endIndex)
		startIndex = endIndex
	# Last split gets all the extra indexes as well
	indices[k] = (indices[k][0], videoLength)
	return {k : np.array(range(indices[k][0], indices[k][1]), dtype=np.uint32) for k in dataSplits}

def computeIndices(dataSplitMode, dataSplits, videoLength, sequenceSize):
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
	dataSplitIndexes = splitFn(videoLength, dataSplits)

	indicesFn = {
		"random" : computeIndicesRandom,
		"sequential" : computeIndicesSequential,
		"sequential_then_random" : computeIndicesRandom
	}[dataSplitMode]
	result = {}
	for split in dataSplitIndexes:
		item = indicesFn(dataSplitIndexes[split], sequenceSize)
		# Since the sequence sizes could mean we have indices outside the possible range, thus why we clip the result.
		result[split] = np.clip(item, 0, videoLength)
	return result