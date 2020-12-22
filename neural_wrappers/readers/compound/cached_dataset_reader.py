# import numpy as np
# from typing import Any
# from pathlib import Path
# from pycache import Cache
# from tqdm import trange
# from .dataset_reader import DatasetReader, DatasetIndex
# from ..utilities import deepCheckEqual
from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from ..batched_dataset_reader import BatchedDatasetReader, BatchedDatasetEpochIterator
from ..compound_batched_dataset_reader import CompoundBatchedDatasetReader
from ..dataset_reader import DatasetReader
from ..dataset_types import *


class CachedBatchedDatasetReader(CompoundBatchedDatasetReader):
	def __init__(self, baseReader:BatchedDatasetReader, cache:Cache, buildCache:bool):
		super().__init__(baseReader)
		self.cache = cache
		self.buildCache = buildCache

	@overrides
	def getBatchItem(self, index):
		assert not isinstance(index, int)

		cacheFile = str(index)
		if self.cache.check(cacheFile):
			item = self.cache.get(cacheFile)
		else:
			item = self.baseReader.getBatchItem(index)
			self.cache.set(cacheFile, item)
		return item

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Cached Batched Dataset Reader]"
		summaryStr += "\n - Path: %s" % self.datasetPath
		summaryStr += "\n - Type: %s" % type(self)
		summaryStr += "\n - Data buckets:"
		for dataBucket in self.datasetFormat.dataBuckets:
			summaryStr += "\n   - %s => %s" % (dataBucket, self.datasetFormat.dataBuckets[dataBucket])
		summaryStr += "\n - Num data: %d. Num batches: %d." % (len(self), len(self.getBatches()))
		summaryStr += "\n - Cache: %s. Build cache: %s" % (self.cache, self.buildCache)
		return summaryStr

# @param[in] baseReader The base dataset reader which is used as composite for caching
# @param[in] cache The PyCache Cache object used for caching purposes
# @param[in] buildCache Whether to do a pass through the entire dataset once before starting the iteration
def CachedDatasetReader(baseReader:Union[DatasetReader, BatchedDatasetReader], cache:Cache, buildCache:bool=False):
	assert baseReader.isCacheable, "Base Reader %s must have the property isCacheble true" % type(baseReader)
	if isinstance(baseReader, BatchedDatasetReader):
		return CachedBatchedDatasetReader(baseReader, cache, buildCache)
	else:
		assert False, "TODO"

# # @brief A composite dataset reader that has a base reader attribute which, will use as a descriptor for caching
# #  purposes during iteration based on its hash (thus the base reader can define its own hash function for better
# #  optimality during different runs). During the first epoch, it will build a dataset cache (no memory constraints
# #  are expected), while the next ones will reuse these cached values for O(1) access.
# # Assumes that baseReader.getBatchdatasetIndex(i, T, B) will always return the same values for any particular
# #  (i, T, B) pairs, thus no stochastic datasets are to be expected.
# class CachedDatasetReader(DatasetReader):
# 	# @param[in] baseReader The base dataset reader which is used as composite for caching
# 	# @param[in] cache The PyCache Cache object used for caching purposes
# 	# @param[in] buildCache Whether to do a pass through the entire dataset once before starting the iteration
# 	#  (this will speed overall running time at the detriment of startup time)
# 	def __init__(self, baseReader:DatasetReader, cache:Cache, buildCache:bool=False):
# 		assert isinstance(baseReader, DatasetReader)
# 		self.baseReader = baseReader
# 		self.cache = cache
# 		self.buildCache = buildCache

# 	def checkAndBuildCache(self, topLevel, batchSize):
# 		N = self.getNumIterations(topLevel, batchSize)
# 		generator = self.baseReader.iterateOneEpoch(topLevel, batchSize)

# 		def buildRegular(generator, N, topLevel, batchSize):
# 			for i in trange(1, N, desc="[CachedDatasetReader] Building (regular)"):
# 				cacheFile = "%s/%d/%d" % (topLevel, batchSize, i)
# 				if self.cache.check(cacheFile):
# 					continue
# 				else:
# 					item = next(generator)
# 					self.cache.set(cacheFile, item)

# 		def buildDirty(generator, N, topLevel, batchSize):
# 			for i in trange(1, N, desc="[CachedDatasetReader] Building (dirty)"):
# 				cacheFile = "%s/%d/%d" % (topLevel, batchSize, i)
# 				item = next(generator)
# 				self.cache.set(cacheFile, item)

# 		# Check first item to decide whether to announce or not
# 		cacheFile = "%s/%d/0" % (topLevel, batchSize)
# 		if not self.cache.check(cacheFile):
# 			print("[CachedDatasetReader] Cache is not built. Building...")
# 			buildRegular(generator, N, topLevel, batchSize)
# 		else:
# 			# Do a consistency check
# 			itemGen = next(generator)
# 			item = self.cache.get(cacheFile)

# 			try:
# 				item = type(itemGen)(item)
# 				dirty = not deepCheckEqual(item, itemGen)
# 			except Exception:
# 				dirty = True

# 			if dirty:
# 				print("[CachedDatasetReader] Cache is dirty. Rebuilding...")
# 				self.cache.set(cacheFile, itemGen)
# 				buildDirty(generator, N, topLevel, batchSize)

# 	@overrides
# 	def getDataset(self, topLevel:str) -> Any:
# 		return self.baseReader.getDataset(topLevel)

# 	@overrides
# 	def getNumData(self, topLevel:str) -> int:
# 		N = self.baseReader.getNumData(topLevel)
# 		return N

# 	# @overrides
# 	# def getBatchDatasetIndex(self, i:int, topLevel:str, batchSize:int) -> DatasetIndex:
# 	# 	return self.baseReader.getBatchDatasetIndex(i, topLevel, batchSize)

# 	@overrides
# 	def iterateOneEpoch(self, topLevel:str, batchSize:int):
# 		if self.buildCache:
# 			self.checkAndBuildCache(topLevel, batchSize)
# 			# The cache should be valid now for this process.
# 			self.buildCache = False

# 		N = self.getNumIterations(topLevel, batchSize)
# 		generator = self.baseReader.iterateOneEpoch(topLevel, batchSize)
# 		for i in range(N):
# 			cacheFile = "%s/%d/%d" % (topLevel, batchSize, i)
# 			if self.cache.check(cacheFile):
# 				item = self.cache.get(cacheFile)
# 			else:
# 				item = next(generator)
# 				self.cache.set(cacheFile, item)
# 			yield item

# 	@overrides
# 	def __str__(self):
# 		Str = "[Cached Dataset Reader]"
# 		Str += "\n - Cache: %s" % str(self.cache)
# 		Str += "\n - Build cache: %s" % self.buildCache
# 		Str += "\n - %s" % str(self.baseReader)
# 		return Str

# 	def __getattr__(self, x):
# 		return getattr(self.baseReader, x)
