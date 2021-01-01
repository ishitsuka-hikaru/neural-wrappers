# Helper class that takes a non-batched dataset reader and makes it batched, by merging multiple items via a merging
#  function that is provided by the user.
from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from collections.abc import Iterable
from typing import Tuple, List
from ..batched_dataset_reader import BatchedDatasetReader
from ..compound_dataset_reader import CompoundDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class MergeBatchedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:DatasetReader, mergeFn:Callable[[List[DatasetItem]], DatasetItem], \
		batchesFn=lambda: BatchedDatasetReader.getBatches(self)):

		try:
			batches = baseReader.getBatches()
			assert False, "Already a batched dataset, sir!"
		except Exception:
			pass
		super().__init__(baseReader)
		self.mergeFn = mergeFn
		self.batchesFn = batchesFn

	def getBatches(self):
		return self.batchesFn()

	# @brief Gets the items of this batch, one by one, from the base reader, and then
	#  merges them together using the provided merge method.
	# @reutrn The current batch of items.
	@overrides
	def __getitem__(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		if isinstance(i, slice):
			i = np.arange(i.start, i.stop)
		if isinstance(i, int):
			i = [i]
		assert isinstance(i, Iterable), "Got type: %s" % type(i)

		items = [self.baseReader[j] for j in i]
		items = self.mergeFn(items)
		return items

	def __str__(self) -> str:
		summaryStr = "[MergeBatchedDatasetReader]"
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr

