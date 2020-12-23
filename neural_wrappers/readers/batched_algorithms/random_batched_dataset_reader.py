from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from ..batched_dataset_reader import BatchedDatasetReader, BatchedDatasetEpochIterator
from ..compound_dataset_reader import CompoundDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class RandomBatchedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader):
		super().__init__(baseReader)
		self.numShuffles = 0

	def getShuffle(self):
		N = self.getNumData()
		S = 0
		batches = []
		while S < N:
			nLeft = N - S
			thisBatch = np.random.randint(1, nLeft + 1)
			S += thisBatch
			batches.append(thisBatch)
		assert sum(batches) == N
		self.numShuffles += 1
		# print("[getShuffle] New shuffle. N=%d. batches=%s. numShuffles=%d" % (len(batches), batches, self.numShuffles))
		# breakpoint()
		return batches

	@overrides
	def getBatches(self) -> List[int]:
		return self.getShuffle()

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		return BatchedDatasetEpochIterator(self)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Random Batched Dataset Reader]"
		summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		summaryStr += "\n - Num data: %d. Num batches this trial: %d. Num shuffles so far: %d" % \
			(len(self), len(self.getBatches()), self.numShuffles)
		return summaryStr
