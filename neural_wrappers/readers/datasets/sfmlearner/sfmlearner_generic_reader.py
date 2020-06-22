from typing import Any
from ...dataset_reader import DatasetReader
from ...internal import DatasetIndex

class SfmLearnerGenericReader(DatasetReader):
	def getDataset(self, topLevel : str) -> Any:
		raise NotImplementedError("Should have implemented this")

	def getNumData(self, topLevel : str) -> int:
		raise NotImplementedError("Should have implemented this")

	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		raise NotImplementedError("Should have implemented this")
