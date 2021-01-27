from overrides import overrides
from typing import List, Tuple
from ..utils import batchIndexFromBatchSizes
from ..batched_dataset_reader import BatchedDatasetReader
from ...compound_dataset_reader import CompoundDatasetReader
from ...dataset_reader import DatasetReader
from ...dataset_types import *

class StaticSizedBatchedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader, batchSize:int):
		super().__init__(baseReader)
		assert batchSize == -1 or batchSize > 0
		self.batchSize = len(self) if batchSize == -1 else batchSize
		self.setBatches()
		self.datasetFormat.isCacheable = True

	# @param[in] batchSize The static batch size required to iterate one epoch. If the batch size is not divisible by
	#  the number of items, the last batch will trimmed accordingly. If the provided value is -1, it is set to the
	#  default value of the entire dataset, based on self.getNumData()
	def setBatches(self):
		N = len(self)
		B = self.batchSize
		n = N // B + (N % B != 0)
		batchLens = n * [B]
		batchLens[-1] -= n * B - N
		self.batchLens = batchLens
		self.batches = batchIndexFromBatchSizes(self.batchLens)

	@overrides
	def getBatches(self) -> List[int]:
		return self.batches

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Static Sized Batched Dataset Reader]"
		summaryStr += "\n %s" % super().__str__()
		summaryStr += "\n - Static batch size: %d" % self.batchSize
		return summaryStr

class StaticBatchedDatasetReader(StaticSizedBatchedDatasetReader): pass