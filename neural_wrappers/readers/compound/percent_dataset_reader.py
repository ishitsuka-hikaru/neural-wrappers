from overrides import overrides
from ..dataset_reader import DatasetReader
from ..compound_dataset_reader import CompoundDatasetReader

# @brief A composite dataset reader that has a base reader attribute which it can partially use based on the percent
#  defined in the constructor
class PercentDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:DatasetReader, percent:float):
		super().__init__(baseReader)
		assert percent > 0 and percent <= 100
		self.percent = percent

	@overrides
	def getBatches(self):
		batches = super().getBatches()
		N = len(batches)
		newN = int(N * self.percent / 100)
		return batches[0 : newN]

	@overrides
	def __len__(self) -> int:
		N = super().__len__()
		return int(N * self.percent / 100)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[PercentDatasetReader]"
		summaryStr += "\n - Percent: %2.2f%%" % self.percent
		summaryStr += "\n %s" % super().__str__()
		return summaryStr