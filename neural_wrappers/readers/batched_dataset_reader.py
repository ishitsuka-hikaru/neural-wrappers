from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List
from .compound_dataset_reader import CompoundDatasetReader
from .dataset_types import *

class BatchedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader):
		super().__init__(baseReader)

	@abstractmethod
	def getBatchSizes(self) -> List[int]:
		pass

	# merge(i1, b1, i2, b2) -> i(1,2)
	@abstractmethod
	def mergeItems(self, item1:Item, batch1:int, item2:Item, batch2:int) -> Item:
		pass

	# split(i(1,2), sz1, sz2) -> i1, i2
	@abstractmethod
	def splitItems(self, item:Item, size1:int, size2:int) -> Tuple[Item, Item]:
		pass

	# @brief Returns the item at index i. Basically g(i) -> Item(i), B(i)
	# @return The item at index i and the batch count of this item (number of items inside the item)
	@overrides
	def getItem(self, i:DatasetIndex) -> Tuple[Item, int]:
		item = self.baseReader.getItem(i)
		b = self.baseReader.getBatchSizes()[i]
		return item, b
