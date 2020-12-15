from overrides import overrides
from typing import Iterator, Dict, List, Callable, Union, Optional
from .batched_dataset_reader import BatchedDatasetReader, DimGetterCallable

class StaticBatchedDatasetReader(BatchedDatasetReader):
	def __init__(self, dataBuckets:Dict[str, List[str]], dimGetter:Dict[str, DimGetterCallable], \
		dimTransform:Dict[str, Dict[str, Callable]], batchSize:Optional[Union[int, Dict[str, int]]]=None):
		super().__init__(dataBuckets, dimGetter, dimTransform)
		self.setBatchSize(batchSize)
	
	def setBatchSize(self, batchSize):
		self.batchSize = batchSize

	@overrides
	def getBatchSize(self, topLevel:str, i:int=0):
		if isinstance(self.batchSize, int):
			N = self.batchSize
		elif isinstance(self.batchSize, dict):
			N = self.batchSize[topLevel]

		if N == -1:
			N = self.getNumData(topLevel)
		return N

	@overrides
	def getNumIterations(self, topLevel:str) -> int:
		N = self.getNumData(topLevel)
		B = self.getBatchSize(topLevel)
		n = N // B + (N % B != 0)
		return n

	@overrides
	def summary(self) -> str:
		summaryStr = super().summary()
		summaryStr += " - Static batch size: %d" % self.batchSize
		return summaryStr