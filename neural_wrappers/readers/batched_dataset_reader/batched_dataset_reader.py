from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from typing import List, Dict, Any, Iterator
from .utils import getBatchLens
from ..dataset_reader import DatasetReader, DatasetEpochIterator
from ..dataset_types import *

class BatchedDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:BatchedDatasetReader):
		# Each iterator has it's own batches (can change for some readers, such as RandomBatchedDatasetReader, where
		#  each epoch has its own set of batches).
		super().__init__(reader)
		try:
			self.batches = reader.getBatches()
			self.batchLens = getBatchLens(self.batches)
			self.len = len(self.batches)
		except Exception:
			# Must be updated from somewhere else (i.e. CompoundDatasetEpochIterator)
			self.batches = None
			self.batchLens = None
			self.len = None

	@overrides
	def __len__(self):
		assert not self.batches is None, "Must be set before calling iterate()/iterateOneEpoch()"
		return super().__len__()

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
	def __getitem__(self, ix):
		# assert not self.batches is None, "Batches must be computed before accessing data."
		# batchIndex = self.batches[ix]
		# batchSize = self.batchLens[ix]
		# batchItem = super().__getitem__(batchIndex)
		# return batchItem, batchSize
		ix = (ix, self.batches, self.batchLens)
		return self.reader[ix]

class BatchedDatasetReader(DatasetReader):
	def getBatches(self) -> List[int]:
		raise NotImplementedError("Must be implemented by the reader!")

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		return BatchedDatasetEpochIterator(self)

	@overrides
	def __getitem__(self, index:DatasetIndex) -> DatasetItem:
		ix, batches, batchLens = index
		batchIndex = batches[ix]
		batchSize = batchLens[ix]
		batchItem = super().__getitem__(batchIndex)
		return batchItem, batchSize

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
