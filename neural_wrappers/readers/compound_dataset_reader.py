from overrides import overrides
from typing import List
from .dataset_reader import DatasetReader
from .batched_dataset_reader import BatchedDatasetReader
from .dataset_types import *

# Helper class for batched algorithms (or even more (?))
class CompoundDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:DatasetReader):
		assert isinstance(baseReader, DatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader

	def getBatchIndex(self, batches, index):
		return self.baseReader.getBatchIndex(batches, index)

	def getBatches(self):
		return self.baseReader.getBatches()

	@overrides
	def iterateOneEpoch(self):
		if not hasattr(self.baseReader, "getBatches"):
			return DatasetReader.iterateOneEpoch(self)
		return super().iterateOneEpoch()

	@overrides
	def getDataset(self):
		return self.baseReader.getDataset()

	@overrides
	def __len__(self):
		return len(self.baseReader)

	@overrides
	def __getitem__(self, key):
		return self.baseReader.__getitem__(key)

	def __getattr__(self, key):
		return getattr(self.baseReader, key)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[CompoundDatasetReader]"
		summaryStr += "\n - Type: %s" % type(self.baseReader)
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr
