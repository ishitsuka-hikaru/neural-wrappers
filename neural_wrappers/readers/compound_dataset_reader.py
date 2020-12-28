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
			self.batchLens = getBatchLens(self.batches)
			self.len = len(self.batches)
			self.batchIndexFn = lambda index : self.batches[index]
			self.returnFn = lambda index, batchIndex: (self.reader[batchIndex], self.batchLens[index])

		except Exception:
			self.batches = None
			self.batchLens = None
			self.len = len(reader)
			self.batchIndexFn = lambda index : index
			self.returnFn = lambda index, batchIndex : self.reader[index]

	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			index = self.getIndexMapping(self.ix)
			batchIndex = self.batchIndexFn(index)
			item = self.returnFn(index, batchIndex)
			return item
		raise StopIteration

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
