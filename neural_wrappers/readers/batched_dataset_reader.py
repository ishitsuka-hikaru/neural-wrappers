from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List, Dict, Any, Iterator
from .dataset_reader import DatasetReader
from .dataset_types import *

class BatchedDatasetIterator:
	def __init__(self, reader:BatchedDatasetIterator):
		self.reader = reader
		self.ix = -1
		# Each iterator hgas it's own batches (can change for some readers, such as RandomBatchedDatasetReader, where
		#  each epoch has its own set of batches).
		self.batches = self.reader.getBatches()
		self.len = self.reader.getNumIterations()
		assert self.len == len(self.batches)

	def __len__(self):
		return self.len

	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			index = self.reader.getBatchIndex(self.batches, self.ix)
			batchItem = self.reader.getBatchItem(index)
			return batchItem, self.batches[self.ix]
		raise StopIteration

	def __iter__(self):
		return self

class BatchedDatasetReader(DatasetReader):
	@abstractmethod
	def getBatches(self) -> List[int]:
		pass

	@overrides
	def getNumIterations(self) -> int:
		return len(self.getBatches())

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

	# TODO: Specify this is a _batched index_
	def getBatchItem(self, index:DatasetIndex) -> DatasetItem:
		return super().getItem(index)

	# @brief Returns the item at index i. Basically g(i) -> Item(i), B(i)
	# @return The item at index i and the batch count of this item (number of items inside the item)
	# @overrides
	def getItem(self, i:int) -> Tuple[DatasetItem, int]:
		batches = self.getBatches()
		index = self.getBatchIndex(batches, i)
		batchItem = self.getBatchItem(index)
		B = batches[i]
		return batchItem, B

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
		summaryStr += "\n - Num data: %d. Num batches: %d. Num iterations this epoch: %d" % \
			(self.getNumData(), len(self.getBatchSizes()), self.getNumIterations())
		return summaryStr
