from overrides import overrides
from typing import List, Tuple
from ..batched_dataset_reader import BatchedDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class StaticBatchedDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader, batchSize:int):
		assert isinstance(baseReader, BatchedDatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader
		self.setBatchSize(batchSize)
		self.baseReader.getBatchSizes = self.getBatchSizes

	# @param[in] batchSize The static batch size required to iterate one epoch. If the batch size is not divisible by
	#  the number of items, the last batch will trimmed accordingly. If the provided value is -1, it is set to the
	#  default value of the entire dataset, based on self.getNumData()
	def setBatchSize(self, batchSize:int):
		assert batchSize == 1 or batchSize > 0
		N = self.getNumData()
		if batchSize == -1:
			batchSize = N
		n = N // batchSize
		batchSizes = n * [batchSize]
		if N % batchSize != 0:
			batchSizes.append(N % batchSize)
		self.batchSize = batchSize
		self.batchSizes = batchSizes

	@overrides
	def getBatchSizes(self) -> List[int]:
		return self.batchSizes

	@overrides
	def getDataset(self):
		return self.baseReader

	@overrides
	def getNumData(self):
		return self.baseReader.getNumData()

	# @brief Gets the items of this batch, one by one, from the base reader, and then
	#  merges them together using the provided merge method.
	# @reutrn The current batch of items.
	@overrides
	def getBatchItem(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		return self.baseReader.getBatchItem(i)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Static Batched Dataset Reader]"
		summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		summaryStr += "\n - Num data: %d. Num batches: %d. Num iterations this epoch: %d" % \
			(self.getNumData(), len(self.getBatchSizes()), self.getNumIterations())
		summaryStr += "\n - Static batch size: %d" % self.batchSize
		return summaryStr
