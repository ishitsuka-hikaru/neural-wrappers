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
		cumsum = np.insert(np.cumsum(batches), 0, 0)
		# i = 2 => B = [6, 7, 8, 9]
		# batchIndex = np.arange(cumsum[i], cumsum[i + 1])
		try:
			batchIndex = slice(cumsum[i], cumsum[i + 1])
		except Exception:
			breakpoint()
		return batchIndex

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		batchSizes = self.getBatchSizes()
		for i, (item, b) in enumerate(super().iterateOneEpoch()):
			item, b = self.getItem(i)
			assert b == batchSizes[i]
			yield item, b

	@overrides
	def getNumIterations(self) -> int:
		return len(self.getBatchSizes())

	# @brief Returns the item at index i. Basically g(i) -> Item(i), B(i)
	# @return The item at index i and the batch count of this item (number of items inside the item)
	@overrides
	def getItem(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		item = super().getItem(i)
		b = self.getBatchSizes()[i]
		return item, b

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
