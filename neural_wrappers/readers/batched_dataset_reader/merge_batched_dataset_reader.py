# Helper class that takes a non-batched dataset reader and makes it batched, by merging multiple items via a merging
#  function that is provided by the user.
from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from collections.abc import Iterable
from typing import Tuple, List
from ..batched_dataset_reader import BatchedDatasetReader
from ..compound_dataset_reader import CompoundDatasetReader, CompoundDatasetEpochIterator
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class MergeBatchedDatasetEpochIterator(CompoundDatasetEpochIterator):
	@overrides
	def __getitem__(self, ix):
		index = self.indexFn(ix)
		if isinstance(index, slice):
			index = np.arange(index.start, index.stop)
		if isinstance(index, (int, np.integer)):
			index = [index]
		assert isinstance(index, Iterable), "Got type: %s" % type(index)

		listItems = [self.baseIterator[j] for j in index]
		assert len(listItems) == self.batchLens[ix]
		items = self.reader.mergeFn(listItems)
		return items, len(listItems)

class MergeBatchedDatasetReader(CompoundDatasetReader):
	def __init__(self, baseReader:DatasetReader, mergeFn:Callable[[List[DatasetItem]], DatasetItem], batchesFn=None):
		try:
			batches = baseReader.getBatches()
			alreadyBatched = True
		except Exception as e:
			alreadyBatched = False
		assert not alreadyBatched, "[MergeBatchDatasetReader] Already a batched dataset, sir!"
		super().__init__(baseReader)

		if batchesFn is None:
			def f():
				assert False, "Must be provided or overridden somewhere"
			batchesFn = f

		self.mergeFn = mergeFn
		self.batchesFn = batchesFn

	def getBatches(self):
		return self.batchesFn()

	def iterateOneEpoch(self):
		return MergeBatchedDatasetEpochIterator(self)

	def __len__(self):
		ret = super().__len__()
		return ret

	def __str__(self) -> str:
		summaryStr = "[MergeBatchedDatasetReader]"
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr

