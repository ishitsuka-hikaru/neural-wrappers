from overrides import overrides
from typing import Any, Tuple
from .dataset_reader import DatasetReader
from .dataset_types import *

class CompoundDatasetReader(DatasetReader):
	def __init__(self, baseReader:DatasetReader):
		self.baseReader = baseReader

	def __getattr__(self, key):
		return getattr(self.baseReader, key)

	@overrides
	def getDataset(self) -> Any:
		return self.baseReader.getDataset()

	@overrides
	def getNumData(self) -> int:
		return self.baseReader.getNumData()

	# @brief Returns the item at index i. Basically g(i) -> Item(i), B(i)
	# @return The item at index i and the batch count of this item (number of items inside the item)
	@overrides
	def getIndex(self, i:int) -> DatasetIndex:
		return self.baseReader.getIndex(i)