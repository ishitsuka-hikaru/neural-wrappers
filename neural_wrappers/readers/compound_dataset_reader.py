from __future__ import annotations
from overrides import overrides
from .dataset_reader import DatasetReader, DatasetEpochIterator
from .dataset_types import *

class CompoundDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:DatasetReader):
		assert isinstance(reader, DatasetReader)
		super().__init__(reader)

		try:
			from .batched_dataset_reader.utils import getBatchLens
			self.batches = reader.getBatches()
			self.len = len(self.batches)
			self.batchLens = getBatchLens(self.batches)
			self.getFn = self.batchedGetFn
		except Exception:
			self.len = len(reader)
			self.getFn = self.regularGetFn

	def batchedGetFn(self, ix):
		batchIndex = self.batches[ix]
		batchLen = self.batchLens[ix]
		batchItem = self.reader[batchIndex]
		return batchItem, batchLen

	def regularGetFn(self, ix):
		item = self.reader[ix]
		return item

	@overrides
	def __getitem__(self, ix):
		return self.getFn(ix)

# Helper class for batched algorithms (or even more (?))
class CompoundDatasetReader(DatasetReader):
	def __init__(self, baseReader:DatasetReader):
		assert isinstance(baseReader, DatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader

	def getBatches(self):
		return self.baseReader.getBatches()

	@overrides
	def iterateOneEpoch(self):
		return CompoundDatasetEpochIterator(self)

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
