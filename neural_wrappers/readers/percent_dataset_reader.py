from .dataset_reader import DatasetReader, DatasetIndex
from overrides import overrides
from typing import Any

# @brief A composite dataset reader that has a base reader attribute which it can partially use based on the percent
#  defined in the constructor
class PercentDatasetReader(DatasetReader):
	def __init__(self, baseReader:DatasetReader, percent:float):
		assert percent > 0 and percent <= 100
		self.baseReader = baseReader
		self.percent = percent

	def __getattr__(self, x):
		return getattr(self.baseReader, x)

	@overrides
	def getDataset(self, topLevel:str) -> Any:
		return self.baseReader.getDataset(topLevel)

	@overrides
	def getNumData(self, topLevel:str) -> int:
		N = self.baseReader.getNumData(topLevel)
		return N * self.percent // 100

	@overrides
	def getBatchDatasetIndex(self, i:int, topLevel:str, batchSize:int) -> DatasetIndex:
		return self.baseReader.getBatchDatasetIndex(i, topLevel, batchSize)
