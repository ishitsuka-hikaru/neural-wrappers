import numpy as np
from overrides import overrides
from ..dataset_reader import DatasetReader, DatasetEpochIterator
from ..compound_dataset_reader import CompoundDatasetReader
# from ..batched_dataset_reader.batched_dataset_reader import BatchedDatasetEpochIterator

class RandomIndexDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:DatasetReader):
		super().__init__(reader)
		self.reader.permutation = np.random.permutation(len(self))

# @brief A composite dataset reader that has a base reader attribute which it can partially use based on the percent
#  defined in the constructor
class RandomIndexDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:DatasetReader, seed:int):
		super().__init__(baseReader)
		np.random.seed(seed)
		self.seed = seed
		self.permutation = None

	@overrides
	def iterateOneEpoch(self):
		return RandomIndexDatasetEpochIterator(self.baseReader)

	@overrides
	def __getitem__(self, ix):
		# print("HERE??")
		# breakpoint()
		assert not self.permutation is None, "Call iterateOneEpoch first to generate a permutation."
		index = self.permutation[ix]
		return super().__getitem__(index)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[RandomIndexDatasetReader]"
		summaryStr += "\n - Seed: %d" % self.seed
		summaryStr += "\n %s" % super().__str__()
		return summaryStr