import numpy as np
from typing import Any, Dict, Callable, List
from ...dataset_reader import DatasetReader, DimGetterCallable
from ...internal import DatasetIndex

class SfmLearnerGenericReader(DatasetReader):
	def __init__(self, dataBuckets : Dict[str, List[str]], dimGetter : Dict[str, DimGetterCallable], \
		dimTransform:Dict[str, Dict[str, Callable]], sequenceSize:int, dataSplits:Dict[str, float], \
		intrinsicMatrix:np.ndarray = np.eye(3)):
		super().__init__(dataBuckets, dimGetter, dimTransform)
		assert sequenceSize > 1
		assert sum(dataSplits.values()) == 1

		self.intrinsicMatrix = intrinsicMatrix
		self.sequenceSize = sequenceSize
		self.dataSplits = dataSplits

	def getDataset(self, topLevel : str) -> Any:
		raise NotImplementedError("Should have implemented this")

	def getNumData(self, topLevel : str) -> int:
		raise NotImplementedError("Should have implemented this")

	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		raise NotImplementedError("Should have implemented this")
