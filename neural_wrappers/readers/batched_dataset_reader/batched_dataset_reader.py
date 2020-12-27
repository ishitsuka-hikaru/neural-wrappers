from __future__ import annotations
from overrides import overrides
from abc import abstractmethod
from typing import List, Dict, Any, Iterator
from ..dataset_reader import DatasetReader
from ..dataset_types import *

class BatchedDatasetReader(DatasetReader):
	def getBatches(self) -> List[int]:
		raise NotImplementedError("Must be implemented by the reader!")

	@overrides
	def iterateOneEpoch(self) -> Iterator[Dict[str, Any]]:
		from .batched_dataset_epoch_iterator import BatchedDatasetEpochIterator
		return BatchedDatasetEpochIterator(self)

	@overrides
	def __getitem__(self, index:DatasetIndex) -> DatasetItem:
		assert not isinstance(index, int)
		return super().__getitem__(index)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Batched Dataset Reader]"
		# summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		try:
			numBatches = "%d" % len(self.getBatches())
		except Exception:
			numBatches = "Not implemented"
		summaryStr += "\n - Num data: %d. Num batches: %s." % (len(self), numBatches)
		return summaryStr
