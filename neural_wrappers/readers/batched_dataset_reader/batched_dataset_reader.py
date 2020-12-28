from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from typing import List, Dict, Any, Iterator
from ..dataset_reader import DatasetReader, DatasetEpochIterator
from ..dataset_types import *

class BatchedDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:BatchedDatasetReader):
		assert isinstance(reader, BatchedDatasetReader)
		super().__init__(reader)
		# Each iterator hgas it's own batches (can change for some readers, such as RandomBatchedDatasetReader, where
		#  each epoch has its own set of batches).
		self.batches = self.reader.getBatches()
		self.batchLens = self.getBatchLens()
		self.len = len(self.batches)

	def getBatchLens(self):
		if isinstance(self.batches[0], slice):
			def sliceLen(batchIndex):
				step = batchIndex.step if not batchIndex.step is None else 1
				N = batchIndex.stop - batchIndex.start
				B = N // step + (N % step != 0)
				return B
			return [sliceLen(x) for x in self.batches]
		elif hasattr(self.batches[0], "len"):
			return [len(x) for x in self.batches]
		else:
			assert False, "Provide a way to find length of batches... Type: %s" % type(self.batches[0])

	# We update the logic of getting as follows. For plain (non-batched) datasets we had
	#  - items = reader[mapping(ix)] for ix in [0 : len(self) - 1]
	# However, we are in a batched situation, so for each index, we are receiving a batch of items
	# batches = [ B1, B1, B1, B2, B2, B3, B4, B4, B5] where B1 is the batch for index 1 (len 3), B2 for index 2 etc.
	# We are still passing a routing table, however this routing table is for batch indexes.
	# Thus, for this epoch we have a list of batches, which can be anything iterable:
	#  batches = reader.getBatches() and batches[ix] = the ixth batch, where ix is a number going from
	#  [0 : len(batches) - 1]. What we get from batches[ix] is an access operator, which is understood by the underyling
	#  __getitem__/dimGetter structure. It can be a list of raw indexes, a slice, a range etc.
	#  We are also passing through the mapping, so index = mapping(ix) => batchItem = reader[index]
	#    index = mapping(ix)
	@overrides
	def __next__(self):
		self.ix += 1
		if self.ix < self.len:
			index = self.getIndexMapping(self.ix)
			batchIndex = self.batches[index]
			batchSize = self.batchLens[index]
			batchItem = self.reader[batchIndex]
			return batchItem, batchSize
		raise StopIteration

class BatchedDatasetReader(DatasetReader):
	def getBatches(self) -> List[int]:
		raise NotImplementedError("Must be implemented by the reader!")

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		return BatchedDatasetEpochIterator(self)

	@overrides
	def __getitem__(self, index:DatasetIndex) -> DatasetItem:
		assert not isinstance(index, int)
		return super().__getitem__(index)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Batched Dataset Reader]"
		# summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		try:
			numBatches = "%d" % len(self.getBatches())
		except Exception:
			numBatches = "Not implemented"
		summaryStr += "\n - Num data: %d. Num batches: %s." % (len(self), numBatches)
		return summaryStr
