from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List
from ..batched_dataset_reader import BatchedDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class MergeBatchedDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:DatasetReader):
		self.baseReader = baseReader

	# merge(i1, b1, i2, b2) -> i(1,2)
	@abstractmethod
	def mergeItems(self, item:List[DatasetItem], batchSize:int) -> DatasetItem:
		pass

	@overrides
	def getDataset(self):
		return self.baseReader

	@overrides
	def getNumData(self):
		return self.baseReader.getNumData()

	@overrides
	def getIndex(self, i:int) -> DatasetIndex:
		batchIndex = super().getIndex(i)
		if isinstance(batchIndex, slice):
			batchIndex = np.arange(batchIndex.start, batchIndex.stop)
		return batchIndex

	# @brief Gets the items of this batch, one by one, from the base reader, and then
	#  merges them together using the provided merge method.
	# @reutrn The current batch of items.
	@overrides
	def getItem(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		batchIndex = self.getIndex(i)
		items = [self.baseReader.getItem(j) for j in batchIndex]
		b = len(items)
		items = self.mergeItems(items, b)
		return items, b
