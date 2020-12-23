from overrides import overrides
from typing import List
from .dataset_reader import DatasetReader
from .batched_dataset_reader import BatchedDatasetReader
from .dataset_types import *

# Helper class for batched algorithms (or even more (?))
class CompoundDatasetReader(DatasetReader):
	def __init__(self, baseReader:DatasetReader):
		assert isinstance(baseReader, DatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader

	@overrides
	def getDataset(self):
		return self.baseReader.getDataset()

	@overrides
	def getNumData(self):
		return self.baseReader.getNumData()

	@overrides
	def __getitem__(self, key):
		return self.baseReader[key]

	def __getattr__(self, key):
		return getattr(self.baseReader, key)

	def __str__(self) -> str:
		summaryStr = "[CompoundDatasetReader]"
		summaryStr += "\n - Type: %s" % type(self.baseReader)
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr

class CompoundBatchedDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader):
		assert isinstance(baseReader, BatchedDatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader

	@overrides
	def getBatchIndex(self, batches, index):
		return self.baseReader.getBatchIndex(batches, index)

	@overrides
	def getBatches(self):
		return self.baseReader.getBatches()

	@overrides
	def getDataset(self):
		return self.baseReader.getDataset()

	@overrides
	def getNumData(self):
		return self.baseReader.getNumData()

	def __getattr__(self, key):
		return getattr(self.baseReader, key)

	@overrides
	def __getitem__(self, key):
		return self.baseReader[key]

	def __str__(self) -> str:
		summaryStr = "[CompoundBatchedDatasetReader]"
		summaryStr += "\n - Type: %s" % type(self.baseReader)
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr