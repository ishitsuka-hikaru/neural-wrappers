from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from tqdm import trange
from ..compound_dataset_reader import CompoundDatasetReader, CompoundDatasetEpochIterator
from ..dataset_reader import DatasetReader
from ..batched_dataset_reader import BatchedDatasetReader
from ..dataset_types import *
from ...utilities import deepCheckEqual

class CachedDatasetEpochIterator(CompoundDatasetEpochIterator):
	@overrides
	def __getitem__(self, ix):
		index = self.indexFn(ix)
		key = self.reader.cacheKey(index)
		if self.reader.cache.check(key):
			return self.reader.cache.get(key)
		else:
			item = self.baseIterator.__getitem__(ix)
			self.reader.cache.set(key, item)
			return item

class CachedDatasetReader(CompoundDatasetReader):
	# @param[in] baseReader The base dataset reader which is used as composite for caching
	# @param[in] cache The PyCache Cache object used for caching purposes
	# @param[in] buildCache Whether to do a pass through the entire dataset once before starting the iteration
	def __init__(self, baseReader:DatasetReader, cache:Cache, buildCache:bool=True):
		super().__init__(baseReader)
		assert baseReader.datasetFormat.isCacheable == True, "%s is not cacheable!" % type(baseReader)
		self.cache = cache
		self.buildCache = buildCache

		if self.buildCache:
			self.doBuildCache()

	def iterateOneEpoch(self):
		return CachedDatasetEpochIterator(self)

	def buildRegular(self, iterator):
		for _ in trange(len(iterator), desc="[CachedDatasetReader] Building regular"):
			_ = next(iterator)
		
	def buildDirty(self, iterator):
		baseIterator = iterator.baseReader.iterateOneEpoch()
		for i in trange(len(baseIterator), desc="[CachedDatasetReader] Building dirty"):
			key = self.cacheKey(iterator.indexFn(i))
			item = baseIterator[i]
			self.cache.set(key, item)

	def doBuildCache(self):
		iterator = self.iterateOneEpoch()
		baseIterator = iterator.baseReader.iterateOneEpoch()
		# baseIterator = self.baseReader.iterateOneEpoch()

		# Try a random index to see if cache is built at all.
		randomIx = np.random.randint(0, len(iterator))
		key = self.cacheKey(iterator.indexFn(randomIx))
		if not self.cache.check(key):
			self.buildRegular(iterator)
			return

		# Otherwise, check if cache is dirty. 5 iterations _should_ be enough.
		dirty = False
		for i in range(5):
			item = iterator[randomIx]
			itemGen = baseIterator[randomIx]
			try:
				item = type(itemGen)(item)
				dirty = dirty or (not deepCheckEqual(item, itemGen))
			except Exception:
				dirty = True

			if dirty:
				break
			randomIx = np.random.randint(0, len(iterator))
			key = iterator.reader.cacheKey(randomIx)

		if dirty:
			print("[CachedDatasetReader] Cache is dirty. Rebuilding...")
			self.buildDirty(iterator)

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Cached Dataset Reader]"
		summaryStr += "\n - Cache: %s. Build cache: %s" % (self.cache, self.buildCache)
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr
