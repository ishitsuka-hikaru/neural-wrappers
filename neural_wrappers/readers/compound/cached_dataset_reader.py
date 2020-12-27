from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from tqdm import trange
from ..compound_dataset_reader import CompoundDatasetReader
from ..dataset_reader import DatasetReader
from ..batched_dataset_reader import BatchedDatasetReader
from ..batched_dataset_reader.batched_dataset_reader import getBatchIndex
from ..dataset_types import *
from ...utilities import deepCheckEqual

class CachedDatasetReader(CompoundDatasetReader):
	# @param[in] baseReader The base dataset reader which is used as composite for caching
	# @param[in] cache The PyCache Cache object used for caching purposes
	# @param[in] buildCache Whether to do a pass through the entire dataset once before starting the iteration
	def __init__(self, baseReader:DatasetReader, cache:Cache, buildCache:bool=False):
		super().__init__(baseReader)
		self.cache = cache
		self.buildCache = buildCache

		if self.buildCache:
			self.doBuildCache()

	def doBuildCache(self):
		if hasattr(self.baseReader, "getBatches"):
			batches = self.baseReader.getBatches()
			indexFn = lambda i : getBatchIndex(batches, i)
			n = len(batches)
		else:
			indexFn = lambda i : i
			n = len(self.baseReader)

		def buildRegular(reader, n, cache):
			for i in trange(n, desc="[CachedDatasetReader] Building (regular)"):
				index = indexFn(i)
				key = reader.cacheKey(index)
				if cache.check(key):
					continue
				else:
					item = reader[index]
					cache.set(key, item)

		def buildDirty(reader, n, cache):
			for i in trange(1, n, desc="[CachedDatasetReader] Building (dirty)"):
				index = indexFn(i)
				key = reader.cacheKey(index)
				item = reader[index]
				cache.set(key, item)

		index = indexFn(0)
		key = self.baseReader.cacheKey(index)
		if not self.cache.check(key):
			buildRegular(self.baseReader, n, self.cache)
		else:
			# Do a consistency check
			itemGen = self.baseReader[index]
			item = self.cache.get(key)

			try:
				item = type(itemGen)(item)
				dirty = not deepCheckEqual(item, itemGen)
			except Exception:
				dirty = True

			if dirty:
				print("[CachedDatasetReader] Cache is dirty. Rebuilding...")
				self.cache.set(key, itemGen)
				buildDirty(self.baseReader, n, self.cache)

	@overrides
	def __getitem__(self, index):
		cacheFile = self.cacheKey(index)
		if self.cache.check(cacheFile):
			item = self.cache.get(cacheFile)
		else:
			item = super().__getitem__(index)
			self.cache.set(cacheFile, item)
		return item

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Cached Dataset Reader]"
		summaryStr += "\n - Cache: %s. Build cache: %s" % (self.cache, self.buildCache)
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr
