from __future__ import annotations
from overrides import overrides
from .dataset_reader import DatasetReader, DatasetEpochIterator
from .dataset_types import *

class CompoundDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader):
		super().__init__(reader)
		self.baseReader = reader.baseReader

		# Some compound algorithms require to call their iterateOneEpoch method too for initializations, for example
		#  RandomIndexDatasetReader, to initialize the epoch's permutation for __getitem__.
		if isinstance(reader.baseReader, CompoundDatasetReader):
			_ = reader.baseReader.iterateOneEpoch()
		try:
			from .batched_dataset_reader.utils import getBatchLens
			batches = self.reader.getBatches()
			self.batches = batches
			self.batchLens = getBatchLens(batches)
			self.len = len(self.batches)
			self.isBatched = True
			self.indexFn = lambda ix : self.batches[ix]
		except Exception as e:
			self.isBatched = False
			self.len = len(self.reader)
			self.indexFn = lambda ix : ix

	@overrides
	def __getitem__(self, ix):
		index = self.indexFn(ix)
		item = self.reader[index]
		if self.isBatched:
			batchSize = self.batchLens[ix]
			item = item, batchSize
		return item

# Helper class for batched algorithms (or even more (?))
class CompoundDatasetReader(DatasetReader):
	def __init__(self, baseReader:DatasetReader):
		assert isinstance(baseReader, DatasetReader)
		super().__init__(dataBuckets=baseReader.datasetFormat.dataBuckets, \
			dimGetter=baseReader.datasetFormat.dimGetter, dimTransform=baseReader.datasetFormat.dimTransform)
		self.baseReader = baseReader

	# Batched Compound Readers (i.e. MergeBatchedDatasetReader) should update this!
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
