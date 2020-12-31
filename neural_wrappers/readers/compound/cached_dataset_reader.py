from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from tqdm import trange
from ..compound_dataset_reader import CompoundDatasetReader
from ..dataset_reader import DatasetReader
from ..batched_dataset_reader import BatchedDatasetReader
from ..dataset_types import *
from ...utilities import deepCheckEqual

def buildRegular(iterator, cache):
	N = len(iterator)
	for i in trange(N, desc="[CachedDatasetReader] Building regular"):
		key = iterator.reader.cacheKey(i)
		if not cache.check(key):
			item = next(iterator)
			cache.set(key, item)

def buildDirty(iterator, cache):
	N = len(iterator)
	for i in trange(N, desc="[CachedDatasetReader] Building dirty"):
		key = iterator.reader.cacheKey(i)
		# TODO: What about Compound(Compound...)
		item = super(type(iterator.reader), iterator.reader).__getitem__(i)
		cache.set(key, item)

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
		iterator = self.iterateOneEpoch()
		# Try a random index to see if cache is built at all.
		randomIx = np.random.randint(0, len(iterator))
		key = iterator.reader.cacheKey(randomIx)
		if not self.cache.check(key):
			buildRegular(iterator, self.cache)
		else:
			item = self.cache.get(key)
			# What about Compound(Compound...)
			itemGen = super().__getitem__(randomIx)
			try:
				item = type(itemGen)(item)
				dirty = not deepCheckEqual(item, itemGen)
			except Exception:
				dirty = True

			if dirty:
				print("[CachedDatasetReader] Cache is dirty. Rebuilding...")
				# self.cache.set(key, itemGen)
				buildDirty(iterator, self.cache)

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
