from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List, Dict, Any, Iterator
from .dataset_reader import DatasetReader, DatasetIterator
from .dataset_types import *

class BatchedDatasetIterator(DatasetIterator):
	def __init__(self, reader:BatchedDatasetIterator):
		self.reader = reader
		self.ix = -1
		# Each iterator hgas it's own batches (can change for some readers, such as RandomBatchedDatasetReader, where
		#  each epoch has its own set of batches).
		self.batches = self.reader.getBatches()
		self.len = len(self.batches)

	def __len__(self):
		return self.len

	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			index = self.reader.getBatchIndex(self.batches, self.ix)
			batchItem = self.reader.getBatchItem(index)
			return batchItem, self.batches[self.ix]
		raise StopIteration

class BatchedDatasetReader(DatasetReader):
	def getBatches(self) -> List[int]:
		raise NotImplemented("Must be implemented by the reader!")

	def getBatchIndex(self, batches:List[int], i:int) -> DatasetIndex:
		# batches = [1, 5, 4, 2] => cumsum = [0, 1, 6, 10, 12]
		cumsum = np.insert(np.cumsum(batches), 0, 0)
		# i = 2 => B = [6, 7, 8, 9]
		# batchIndex = np.arange(cumsum[i], cumsum[i + 1])
		try:
			batchIndex = slice(cumsum[i], cumsum[i + 1])
		except Exception:
			breakpoint()
		return batchIndex

	def getBatchItem(self, index:DatasetIndex) -> DatasetItem:
		assert not isinstance(index, int)
		return super().getItem(index)

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		return BatchedDatasetIterator(self)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Batched Dataset Reader]"
		summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		summaryStr += "\n - Num data: %d. Num batches: %d." % (len(self), len(self.getBatches()))
		return summaryStr
