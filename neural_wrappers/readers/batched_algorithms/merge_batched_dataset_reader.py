# Helper class that takes a non-batched dataset reader and makes it batched, by merging multiple items via a merging
#  function that is provided by the user.
from overrides import overrides
from abc import abstractmethod
from typing import Tuple, List
# from ..batched_dataset_reader import BatchedDatasetReader
from ..compound_dataset_reader import CompoundDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class MergeBatchedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:DatasetReader):
		try:
			batches = baseReader.getBatches()
			# ix = baseReader.getBatchIndex(batches, 0)
			assert False, "Already a batched dataset, sir!"
		except Exception:
			pass
		super().__init__(baseReader)

		# self.baseReader = baseReader

	# merge(i1, b1, i2, b2) -> i(1,2)
	@abstractmethod
	def mergeItems(self, item:List[DatasetItem]) -> DatasetItem:
		pass

	# @brief Update getBatchIndex as this is by default working with slices.
	# @overrides
	# def getBatchIndex(self, batches:List[int], i:int) -> DatasetIndex:
	# 	cumsum = np.insert(np.cumsum(batches), 0, 0)
	# 	batchIndex = np.arange(cumsum[i], cumsum[i + 1])
	# 	return batchIndex

	# @brief Gets the items of this batch, one by one, from the base reader, and then
	#  merges them together using the provided merge method.
	# @reutrn The current batch of items.
	@overrides
	def __getitem__(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		assert isinstance(i, slice), "Type: %s" % type(i)
		i = np.arange(i.start, i.stop)
		items = [self.baseReader[j] for j in i]
		items = self.mergeItems(items)
		return items

	def __str__(self) -> str:
		summaryStr = "[MergeBatchedDatasetReader]"
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr