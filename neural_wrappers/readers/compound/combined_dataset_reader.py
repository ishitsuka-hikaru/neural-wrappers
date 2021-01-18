import numpy as np
from overrides import overrides
from typing import List
from ..dataset_reader import DatasetReader
from ..compound_dataset_reader import CompoundDatasetReader, CompoundDatasetEpochIterator

class CombinedDatasetReaderIterator(CompoundDatasetEpochIterator):
	def __init__(self, reader):
		self.baseIterators = [CompoundDatasetEpochIterator(baseReader) for baseReader in reader.baseReaders]
		self.ix = -1
		self.Mappings = self.getMappings()
		self.len = len(self)

	def getMappings(self):
		Mappings = np.zeros((len(self), 2), dtype=np.uint32)
		ix = 0
		for i in range(len(self.baseIterators)):
			g = self.baseIterators[i]
			Range = np.arange(len(g))
			Mappings[ix : ix + len(g), 0] = np.repeat([i], len(g))
			Mappings[ix : ix + len(g), 1] = Range
			ix += len(g)
		return Mappings

	@overrides
	def __len__(self) -> bool:
		return sum([len(x) for x in self.baseIterators])

	@overrides
	def __getitem__(self, ix):
		readerIx, readerInnerIx = self.Mappings[ix]
		return self.baseIterators[readerIx].__getitem__(readerInnerIx)

	@overrides
	def __getattr__(self, key):
		X = [getattr(baseIterator, key) for baseIterator in self.baseIterators]
		return X

# @brief A composite dataset reader that has a base reader attribute which it can partially use based on the percent
#  defined in the constructor
class CombinedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReaders:List[DatasetReader]):
		# super().__init__(baseReader)
		assert len(baseReaders) > 1, "Must provide a list of DatasetReaders!"
		firstReader = baseReaders[0]
		assert isinstance(firstReader, DatasetReader)
		for reader in baseReaders[1 : ]:
			assert isinstance(reader, DatasetReader)
			assert reader.datasetFormat == firstReader.datasetFormat, "All readers must provide same DatasetFormat!"

		DatasetReader.__init__(self, dataBuckets=firstReader.datasetFormat.dataBuckets, \
			dimGetter=firstReader.datasetFormat.dimGetter, dimTransform=firstReader.datasetFormat.dimTransform)
		self.baseReaders = [CompoundDatasetReader(reader) for reader in baseReaders]
		self.baseReader = self.baseReaders

	@overrides
	def iterateOneEpoch(self):
		return CombinedDatasetReaderIterator(self)

	@overrides
	def __len__(self):
		return sum([len(reader) for reader in self.baseReaders])

	@overrides
	def __str__(self) -> str:
		summaryStr = "[CombinedDatasetReader]"
		summaryStr += "\n - Num datasets: %d" % (len(self.baseReaders))
		for i, reader in enumerate(self.baseReaders):
			summaryStr += "\n----------- %d -----------" % (i + 1)
			summaryStr += "\n%s" % str(reader)
		return summaryStr