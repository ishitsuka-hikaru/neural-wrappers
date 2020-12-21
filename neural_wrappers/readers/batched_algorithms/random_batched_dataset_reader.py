from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from ..batched_dataset_reader import BatchedDatasetReader, BatchedDatasetIterator
from .compound_batched_dataset_reader import CompoundBatchedDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class RandomBatchedDatasetIterator(BatchedDatasetIterator):
	def __init__(self, reader:RandomBatchedDatasetReader):
		assert isinstance(reader, RandomBatchedDatasetReader)
		self.reader = reader
		self.ix = -1
		# Unique for this epoch!
		self.batches = self.reader.getShuffle()
		self.len = len(self.batches)

class RandomBatchedDatasetReader(CompoundBatchedDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader):
		assert isinstance(baseReader, BatchedDatasetReader)
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
		return batches

	@overrides
	def getBatches(self) -> List[int]:
		return self.getShuffle()

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		return RandomBatchedDatasetIterator(self)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Random Batched Dataset Reader]"
		summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		summaryStr += "\n - Num data: %d. Num batches this trial: %d. Num iterations this epoch: %d" % \
			(self.getNumData(), len(self.getBatchSizes()), self.getNumIterations())
		return summaryStr
