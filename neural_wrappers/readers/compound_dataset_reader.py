from __future__ import annotations
from overrides import overrides
from .dataset_reader import DatasetReader, DatasetEpochIterator
from .dataset_types import *

# class DatasetEpochIterator:
# 	def __init__(self, reader:DatasetReader):
# 		self.reader = reader
# 		self.ix = -1
# 		self.len = len(self.reader)

# 	def __len__(self):
# 		return self.len

# 	def __getitem__(self, ix):
# 		return self.reader[ix]

# 	# The logic of getting an item is. ix is a number going in range [0 : len(self) - 1]. Then, we call dataset's
# 	#  __getitem__ on this. So item = self[index], where __getitem__(ix) = self.reader[ix].
# 	# One-liner: items = self[ix] for ix in [0 : len(self) - 1]
# 	def __next__(self):
# 		self.ix += 1
# 		if self.ix < len(self):
# 			return self.__getitem__(self.ix)
# 		raise StopIteration

# 	def __iter__(self):
# 		return self

class CompoundDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader):
	# 	super().__init__(reader)
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
		self.baseIterator.ix += 1
		if self.baseIterator.ix < len(self.baseIterator):
			return next(self.baseIterator)

	@overrides
	def __getitem__(self, ix):
		index = self.baseIterator.indexFn(ix)
		item = self.baseIterator[index]
		if self.baseIterator.isBatched:
			batchSize = self.baseIterator.batchLens[ix]
			item = item, batchSize
		return item

	@overrides
	def __len__(self):
		return self.baseIterator.__len__()

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
