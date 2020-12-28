from __future__ import annotations
from overrides import overrides
from typing import List
from .dataset_reader import DatasetReader, DatasetEpochIterator
from .batched_dataset_reader import BatchedDatasetReader
from .dataset_types import *

# from .batched_dataset_reader.utils import getBatchIndex

class CompoundDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:DatasetReader):
		assert isinstance(reader, DatasetReader)
		super().__init__(reader)
		try:
			self.batches = reader.getBatches()
			self.len = len(self.batches)
			self.isBatched = True
			self.batchFn = lambda x : getBatchIndex(self.batches, x)
			self.returnFn = lambda index, index2 : (reader[index2], self.batches[index])
		except Exception:
			self.batches = None
			self.isBatched = False
			self.len = len(reader)
			self.batchFn = lambda x : x
			self.returnFn = lambda index, index2 : reader[index2]

	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			index = self.getIndexMapping(self.ix)
			index2 = self.batchFn(index)
			item = self.returnFn(index, index2)
			return item
		raise StopIteration

# Helper class for batched algorithms (or even more (?))
class CompoundDatasetReader(BatchedDatasetReader):
	def __init__(self, baseReader:DatasetReader):
		assert isinstance(baseReader, DatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader

	@overrides
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
