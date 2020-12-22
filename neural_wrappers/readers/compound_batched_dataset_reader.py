from overrides import overrides
from typing import List
from .batched_dataset_reader import BatchedDatasetReader

# Helper class for batched algorithms (or even more (?))
class CompoundBatchedDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader):
		assert isinstance(baseReader, BatchedDatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader

	@overrides
	def getBatches(self) -> List[int]:
		return self.baseReader.getBatches()

	@overrides
	def getDataset(self):
		return self.baseReader.getDataset()

	@overrides
	def getNumData(self):
		return self.baseReader.getNumData()

	def getBatchItem(self, index):
		assert not isinstance(index, int)
		return self.baseReader.getBatchItem(index)

	@overrides
	def getBatchIndex(self, batches, i):
		return self.baseReader.getBatchIndex(batches, i)

	def __getattr__(self, key):
		return getattr(self.baseReader, key)
