from overrides import overrides
from typing import List
from .batched_dataset_reader import BatchedDatasetReader
from .dataset_reader import DatasetReader

class StaticBatchedDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:DatasetReader, batchSize:int):
		super().__init__(baseReader)
		self.batchSize = batchSize

	def setBatchSize(self, batchSize:int):
		self.batchSize = batchSize

	@overrides
	def getBatchSizes(self) -> List[int]:
		res = np.repeat(self.batchSize, self.getNumData())
		return res

	@overrides
	def summary(self) -> str:
		summaryStr = super().summary()
		summaryStr += " - Static batch size: %d" % self.batchSize
		return summaryStr