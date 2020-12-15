import numpy as np
from overrides import overrides
from typing import Iterator, Dict, Union
from abc import abstractmethod

from ..dataset_reader import DatasetReader, DimGetterCallable

class BatchedDatasetReader(DatasetReader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@abstractmethod
	def getBatchSize(self, topLevel:str, i:int):
		pass

	@overrides
	def iterateOneEpoch(self, topLevel:str) -> Iterator[Dict[str, np.ndarray]]:
		dataset = self.getDataset(topLevel)
		N = self.getNumIterations(topLevel)
		for i in range(N):
			# Get the logical index in the dataset
			index = self.getIndex(topLevel, i)
			# Get the data before bucketing, as there could be overlapping items in each bucket, just
			#  normalized differently, so we shouldn't read from disk more times.
			items, copyDims = {}, {}
			for dim in self.allDims:
				items[dim] = self.dimGetter[dim](dataset, index)
				copyDims[dim] = False

			result:Dict[str, np.ndarray] = {}
			# Go through all data buckets (data/labels etc.)
			for dataBucket in self.dataBuckets:
				result[dataBucket] = {}
				# For each bucket, go through all dims (rgb/semantic/depth etc.)
				for dim in self.dataBuckets[dataBucket]:
					item = items[dim]
					# We're making sure that if this item was in other bucket as well, it's copied so we don't alter
					#  same data memory with multiple transforms
					if copyDims[dim]:
						item = item.copy()
					copyDims[dim] = True

					# Apply this item's data transform
					item = self.dimTransform[dataBucket][dim](item)

					# Store it in this batch
					result[dataBucket][dim] = item
			yield result

	@overrides
	def getIndex(self, topLevel:str, i:int):
		batchSize = self.getBatchSize(topLevel, i)
		startIndex = i * batchSize
		endIndex = min((i + 1) * batchSize, self.getNumData(topLevel))
		return range(startIndex, endIndex)
