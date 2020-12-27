from ..dataset_epoch_iterator import DatasetEpochIterator
from .batched_dataset_reader import BatchedDatasetReader
from .utils import getBatchIndex

class BatchedDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:BatchedDatasetReader):
		assert isinstance(reader, BatchedDatasetReader)
		super().__init__(reader)
		# Each iterator hgas it's own batches (can change for some readers, such as RandomBatchedDatasetReader, where
		#  each epoch has its own set of batches).
		self.batches = self.reader.getBatches()
		self.len = len(self.batches)

	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			index = self.getIndexMapping(self.ix)
			index = getBatchIndex(self.batches, index)
			batchItem = self.reader[index]
			# BatchedDatasets return a tuple of (batchItem, batchSize)
			return batchItem, self.batches[self.ix]
		raise StopIteration