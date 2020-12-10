from .dataset_reader import DatasetReader, DatasetIndex
from overrides import overrides
from typing import Any

import numpy as np
from pathlib import Path
from pycache import Cache

# @brief A composite dataset reader that has a base reader attribute which, will use as a descriptor for caching
#  purposes during iteration based on its hash (thus the base reader can define its own hash function for better
#  optimality during different runs). During the first epoch, it will build a dataset cache (no memory constraints
#  are expected), while the next ones will reuse these cached values for O(1) access.
# Assumes that baseReader.getBatchdatasetIndex(i, T, B) will always return the same values for any particular
#  (i, T, B) pairs, thus no stochastic datasets are to be expected.
class CachedDatasetReader(DatasetReader):
	def __init__(self, baseReader:DatasetReader, cache:Cache):
		assert isinstance(baseReader, DatasetReader)
		self.baseReader = baseReader
		self.cache = cache

	def __getattr__(self, x):
		return getattr(self.baseReader, x)

	@overrides
	def getDataset(self, topLevel:str) -> Any:
		return self.baseReader.getDataset(topLevel)

	@overrides
	def getNumData(self, topLevel:str) -> int:
		N = self.baseReader.getNumData(topLevel)
		return N

	@overrides
	def getBatchDatasetIndex(self, i:int, topLevel:str, batchSize:int) -> DatasetIndex:
		return self.baseReader.getBatchDatasetIndex(i, topLevel, batchSize)

	@overrides
	def iterateOneEpoch(self, topLevel:str, batchSize:int):
		N = self.getNumIterations(topLevel, batchSize)
		generator = self.baseReader.iterateOneEpoch(topLevel, batchSize)
		for i in range(N):
			cacheFile = "%s/%d/%d" % (topLevel, batchSize, i)
			if self.cache.check(cacheFile):
				item = self.cache.get(cacheFile)
			else:
				item = next(generator)
				self.cache.set(cacheFile, item)
			yield item

	@overrides
	def __str__(self):
		Str = "[Cached Dataset Reader]"
		Str += "\n - Cache: %s" % str(self.cache)
		Str += "\n - %s" % str(self.baseReader)
		return Str