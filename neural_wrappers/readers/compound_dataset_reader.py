from __future__ import annotations
from overrides import overrides
from .dataset_reader import DatasetReader, DatasetEpochIterator
from .dataset_types import *

class CompoundDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader):
		self.reader = reader
		self.baseReader = reader.baseReader
		self.baseIterator = self.baseReader.iterateOneEpoch()

		try:
			from .batched_dataset_reader.utils import getBatchLens
			batches = self.reader.getBatches()
			self.baseIterator.batches = batches
			self.baseIterator.batchLens = getBatchLens(batches)
			self.baseIterator.len = len(batches)
		except Exception as e:
			pass

		if hasattr(self.baseIterator, "batches"):
			self.baseIterator.isBatched = True
			self.baseIterator.indexFn = lambda ix : self.baseIterator.batches[ix]
		else:
			self.baseIterator.isBatched = False
			self.baseIterator.len = len(self.baseReader)
			self.baseIterator.indexFn = lambda ix : ix


	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			return self.__getitem__(self.ix)
		raise StopIteration

	@overrides
	def __getitem__(self, ix):
		return self.baseIterator.__getitem__(ix)

	@overrides
	def __len__(self):
		return self.len

	def __getattr__(self, key):
		return getattr(self.baseIterator, key)

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
		res = len(self.baseReader)
		return res

	@overrides
	def __getitem__(self, key):
		assert False

	def __getattr__(self, key):
		return getattr(self.baseReader, key)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[CompoundDatasetReader]"
		summaryStr += "\n - Type: %s" % type(self.baseReader)
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr
