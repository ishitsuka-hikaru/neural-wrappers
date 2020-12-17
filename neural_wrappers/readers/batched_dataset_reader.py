from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List, Dict, Any, Iterator
from .dataset_reader import DatasetReader
from .dataset_types import *

class BatchedDatasetReader(DatasetReader):
	@abstractmethod
	def getBatchSizes(self) -> List[int]:
		pass

	@overrides
	def getIndex(self, i:int) -> DatasetIndex:
		# [1, 5, 4, 2]
		batches = self.getBatchSizes()
		# [0, 1, 6, 10, 12]
		cumsum = np.insert(batches.cumsum(), 0, 0)
		# i = 2 => B = [6, 7, 8, 9]
		# batchIndex = np.arange(cumsum[i], cumsum[i + 1])
		batchIndex = slice(cumsum[i], cumsum[i + 1])
		return batchIndex

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		batchSizes = self.getBatchSizes()
		n = len(batchSizes)
		for i in range(n):
			item, b = self.getItem(i)
			assert b == batchSizes[i]
			yield item, b

	# @brief Returns the item at index i. Basically g(i) -> Item(i), B(i)
	# @return The item at index i and the batch count of this item (number of items inside the item)
	@overrides
	def getItem(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		item = super().getItem(i)
		b = self.getBatchSizes()[i]
		return item, b
