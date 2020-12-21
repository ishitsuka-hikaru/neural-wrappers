from overrides import overrides
from typing import List, Tuple
from ..batched_dataset_reader import BatchedDatasetReader
from ..compound_batched_dataset_reader import CompoundBatchedDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class StaticBatchedDatasetReader(CompoundBatchedDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader, batchSize:int):
		assert isinstance(baseReader, BatchedDatasetReader)
		super().__init__(baseReader)
		self.setBatchSize(batchSize)

	# @param[in] batchSize The static batch size required to iterate one epoch. If the batch size is not divisible by
	#  the number of items, the last batch will trimmed accordingly. If the provided value is -1, it is set to the
	#  default value of the entire dataset, based on self.getNumData()
	def setBatchSize(self, batchSize:int):
		assert batchSize == 1 or batchSize > 0
		N = self.getNumData()
		if batchSize == -1:
			batchSize = N
		n = N // batchSize
		batches = n * [batchSize]
		if N % batchSize != 0:
			batches.append(N % batchSize)
		self.batchSize = batchSize
		self.batches = batches

	@overrides
	def getBatches(self) -> List[int]:
		return self.batches

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Static Batched Dataset Reader]"
		summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		summaryStr += "\n - Num data: %d. Num batches: %d." % (self.getNumData(), len(self.getBatches()))
		summaryStr += "\n - Static batch size: %d" % self.batchSize
		return summaryStr
