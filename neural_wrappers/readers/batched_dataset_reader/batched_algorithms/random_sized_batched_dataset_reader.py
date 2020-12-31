from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from ..batched_dataset_reader import BatchedDatasetReader
from ..utils import batchIndexFromBatchSizes
from ...compound_dataset_reader import CompoundDatasetReader
from ...dataset_reader import DatasetReader
from ...dataset_types import *

class RandomSizedBatchedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader):
		super().__init__(baseReader)
		self.numShuffles = 0

	def getShuffle(self):
		N = len(self)
		S = 0
		batchLens = []
		while S < N:
			nLeft = N - S
			thisLen = np.random.randint(1, nLeft + 1)
			S += thisLen
			batchLens.append(thisLen)
		assert sum(batchLens) == N
		self.numShuffles += 1
		# print("[getShuffle] New shuffle. N=%d. batches=%s. numShuffles=%d" % (len(batches), batches, self.numShuffles))
		# breakpoint()
		batches = batchIndexFromBatchSizes(batchLens)
		return batches

	@overrides
	def getBatches(self) -> List[int]:
		return self.getShuffle()

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Random Sized Batched Dataset Reader]"
		summaryStr += "\n %s" % super().__str__()
		summaryStr += "\n Num batches this trial: %d. Num shuffles so far: %d" % \
			(len(self), len(self.getBatches()), self.numShuffles)
		return summaryStr

class RandomBatchedDatasetReader(RandomSizedBatchedDatasetReader): pass