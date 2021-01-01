from __future__ import annotations
from overrides import overrides
from .dataset_reader import DatasetReader, DatasetEpochIterator
# from .batched_dataset_reader.batched_dataset_reader import BatchedDatasetReader
from .dataset_types import *

class CompoundDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader):
		from .batched_dataset_reader.batched_dataset_reader import BatchedDatasetEpochIterator
		from .batched_dataset_reader.utils import getBatchLens
		self.reader = reader
		self.baseReader = reader.baseReader


		# breakpoint()
		# self.baseIterator = self.baseReader.iterateOneEpoch()

		try:
			batches = self.reader.getBatches()
			self.batches = batches
			self.batchLens = getBatchLens(batches)
			self.len = len(self.batches)
			self.isBatched = True
		except Exception as e:
			self.isBatched = False
			self.len = len(self.reader)

		# try:
		# 	_ = self.baseReader.getBatches()
		# 	self.origGetBatches = self.baseReader.getBatches
		# 	self.baseReader.getBatches = self.getBatches
		# 	self.epochIterator = BatchedDatasetEpochIterator
		# except Exception:
		# 	self.epochIterator = DatasetEpochIterator
		# 	if hasattr(self.baseReader, "getBatches"):
		# 		self.origGetBatches = self.baseReader.getBatches
		# self.reader = reader
		self.ix = -1

	@overrides
	def __getitem__(self, ix):
		if self.isBatched:
			batchIndex = self.batches[ix]
			batchSize = self.batchLens[ix]
			batchItem = self.reader[batchIndex]
			item = batchItem, batchSize
		else:
			item = self.reader[ix]
		return item

	@overrides
	def __iter__(self):
		return self

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
