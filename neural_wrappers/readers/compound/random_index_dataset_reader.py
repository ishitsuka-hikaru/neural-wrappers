import numpy as np
from overrides import overrides
from ..dataset_reader import DatasetReader, DatasetEpochIterator
from ..compound_dataset_reader import CompoundDatasetReader

class RandomIndexDatasetEpochIterator(DatasetEpochIterator):
	def __init__(self, reader:DatasetReader):
		super().__init__(reader)
		self.permutation = np.random.permutation(len(self))

	@overrides	
	def getIndexMapping(self, ix):
		return self.permutation[ix]

# @brief A composite dataset reader that has a base reader attribute which it can partially use based on the percent
#  defined in the constructor
class RandomIndexDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:DatasetReader, seed:int):
		super().__init__(baseReader)
		np.random.seed(seed)
		self.seed = seed

	@overrides
	def iterateOneEpoch(self):
		return RandomIndexDatasetEpochIterator(self)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[RandomIndexDatasetReader]"
		summaryStr += "\n - Seed: %d" % self.seed
		summaryStr += "\n %s" % super().__str__()
		return summaryStr