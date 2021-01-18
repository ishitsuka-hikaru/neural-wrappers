from overrides import overrides
from ..dataset_reader import DatasetReader
from ..compound_dataset_reader import CompoundDatasetReader, CompoundDatasetEpochIterator
from typing import List

class CombinedDatasetReaderIterator(CompoundDatasetEpochIterator):
	def __init__(self, reader):
		self.baseIterators = [CompoundDatasetEpochIterator(reader) for reader in reader.baseReaders]

	@overrides
	def __len__(self) -> bool:
		return sum([len(x) for x in self.baseIterators])

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

	@overrides
	def iterateOneEpoch(self):
		return CombinedDatasetReaderIterator(self)

	@overrides
	def getBatches(self):
		pass
		# pass
		# batches = super().getBatches()
		# N = len(batches)
		# newN = int(N * self.percent / 100)
		# return batches[0 : newN]

	@overrides
	def __str__(self) -> str:
		summaryStr = "[CombinedDatasetReader]"
		# summaryStr += "\n - Percent: %2.2f%%" % self.percent
		# summaryStr += "\n %s" % super().__str__()
		return summaryStr