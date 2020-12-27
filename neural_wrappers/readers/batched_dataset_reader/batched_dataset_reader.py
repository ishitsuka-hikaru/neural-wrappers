from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List, Dict, Any, Iterator
from .utils import getBatchIndex
from ..dataset_reader import DatasetReader
from ..dataset_epoch_iterator import DatasetEpochIterator
from ..dataset_types import *

class BatchedDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:BatchedDatasetReader):
		assert isinstance(reader, BatchedDatasetReader)
		super().__init__(reader)
		# Each iterator hgas it's own batches (can change for some readers, such as RandomBatchedDatasetReader, where
		#  each epoch has its own set of batches).
		self.batches = self.reader.getBatches()
		self.len = len(self.batches)

	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			index = getBatchIndex(self.batches, self.ix)
			batchItem = self.reader[index]
			# BatchedDatasets return a tuple of (batchItem, batchSize)
			return batchItem, self.batches[self.ix]
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
