import numpy as np
from typing import Any, Dict, Callable, List
from ...dataset_reader import DatasetReader, DimGetterCallable
from ...internal import DatasetIndex

class SfmLearnerGenericReader(DatasetReader):
	def __init__(self, dataBuckets : Dict[str, List[str]], dimGetter : Dict[str, DimGetterCallable], \
		dimTransform:Dict[str, Dict[str, Callable]], dataSplitIndices:Dict[str, np.ndarray], \
		intrinsicMatrix:np.ndarray = np.eye(3)):
		super().__init__(dataBuckets, dimGetter, dimTransform)

		self.intrinsicMatrix = intrinsicMatrix
		self.dataSplitIndices = dataSplitIndices

		# TODO: When we add optical flow, this must be computed somehow else (perhaps a dataSplitIndices object)
		firstItem = dataSplitIndices[list(dataSplitIndices.keys())[0]]
		self.sequenceSize = firstItem.shape[1]
		self.skipFrames = firstItem[0, 1] - firstItem[0, 0]

	def getDataset(self, topLevel : str) -> Any:
		raise NotImplementedError("Should have implemented this")

	def getNumData(self, topLevel : str) -> int:
		raise NotImplementedError("Should have implemented this")

	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		raise NotImplementedError("Should have implemented this")
