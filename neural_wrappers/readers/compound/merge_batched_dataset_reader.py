# Helper class that takes a non-batched dataset reader and makes it batched, by merging multiple items via a merging
#  function that is provided by the user.
from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List
from ..batched_dataset_reader import BatchedDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class MergeBatchedDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:DatasetReader):
		assert not isinstance(baseReader, BatchedDatasetReader), "Already a batched dataset, sir!"
		self.baseReader = baseReader

	# merge(i1, b1, i2, b2) -> i(1,2)
	@abstractmethod
	def mergeItems(self, item:List[DatasetItem]) -> DatasetItem:
		pass

	@overrides
	def getDataset(self):
		return self.baseReader.getDataset()

	@overrides
	def getNumData(self):
		return self.baseReader.getNumData()

	@overrides
	def getBatchIndex(self, batches:List[int], i:int) -> DatasetIndex:
		batchIndex = super().getBatchIndex(batches, i)
		batchIndex = np.arange(batchIndex.start, batchIndex.stop)
		return batchIndex

	# @brief Gets the items of this batch, one by one, from the base reader, and then
	#  merges them together using the provided merge method.
	# @reutrn The current batch of items.
	@overrides
	def __getitem__(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		assert isinstance(i, np.ndarray), "Type: %s" % type(i)
		items = [self.baseReader[j] for j in i]
		items = self.mergeItems(items)
		return items

	def __getattr__(self, key):
		return getattr(self.baseReader, key)