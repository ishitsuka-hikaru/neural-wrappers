from __future__ import annotations
from overrides import overrides
from typing import List, Tuple
from tqdm import trange
from ..compound_dataset_reader import CompoundDatasetReader, CompoundDatasetEpochIterator
from ..dataset_reader import DatasetReader
from ..batched_dataset_reader import BatchedDatasetReader
from ..dataset_types import *
from ...utilities import deepCheckEqual

# def buildRegular(baseIterator, iterator, cache):
# 	N = len(iterator)
# 	# print("REG B4", cache.cache.keys())
# 	for i in trange(N, desc="[CachedDatasetReader] Building regular"):
# 		key = iterator.reader.cacheKey(i)
# 		print(i, key)
# 		if not cache.check(key):
# 			# print(i, cache.cache.keys())
# 			item = baseIterator[i]
# 			# breakpoint()
# 			# print(i, cache.cache.keys())
# 			cache.set(key, item)
# 			# print(i, cache.cache.keys())
# 	print("REG", cache.cache.keys())
# 	# exit()

# def buildDirty(baseIterator, iterator, cache):
# 	N = len(iterator)
# 	print("DRT B4", cache.cache.keys())
# 	for i in trange(N, desc="[CachedDatasetReader] Building dirty"):
# 		key = iterator.reader.cacheKey(i)
# 		assert key in cache.cache
# 		oldItem = cache.get(key)
# 		item = baseIterator[i]
# 		# item = super(type(iterator.reader), iterator.reader).__getitem__(i)
# 		cache.set(key, item)
# 		newItem = cache.get(key)
# 	print("DRT", cache.cache.keys())

class CachedDatasetEpochIterator(CompoundDatasetEpochIterator):
	def __init__(self, reader):
		super().__init__(reader)
		# self.cache = self.reader.cache
	
	@overrides
	def __getitem__(self, ix):
		# key = self.reader.cacheKey(ix)
		index = self.indexFn(ix)
		key = self.reader.cacheKey(index)
		if self.reader.cache.check(key):
			return self.reader.cache.get(key)
		else:
			item = super().__getitem__(ix)
			self.reader.cache.set(key, item)
			return item

class CachedDatasetReader(CompoundDatasetReader):
	# @param[in] baseReader The base dataset reader which is used as composite for caching
	# @param[in] cache The PyCache Cache object used for caching purposes
	# @param[in] buildCache Whether to do a pass through the entire dataset once before starting the iteration
	def __init__(self, baseReader:DatasetReader, cache:Cache, buildCache:bool=True):
		super().__init__(baseReader)
		self.cache = cache
		self.buildCache = buildCache

		# if self.buildCache:
		# 	self.doBuildCache()

	def iterateOneEpoch(self):
		return CachedDatasetEpochIterator(self)


	# def doBuildCache(self):
	# 	iterator = self.iterateOneEpoch()
	# 	baseIterator = self.baseReader.iterateOneEpoch()

	# 	# Try a random index to see if cache is built at all.
	# 	randomIx = np.random.randint(0, len(iterator))
	# 	key = iterator.reader.cacheKey(randomIx)
	# 	if not self.cache.check(key):
	# 		buildRegular(baseIterator, iterator, self.cache)
	# 		return

	# 	# Otherwise, check if cache is dirty. 5 iterations _should_ be enough.
	# 	dirty = False
	# 	for i in range(5):
	# 		item = self.cache.get(key)
	# 		itemGen = baseIterator[randomIx]

	# 		try:
	# 			item = type(itemGen)(item)
	# 			dirty = dirty or (not deepCheckEqual(item, itemGen))
	# 		except Exception:
	# 			dirty = True

	# 		if dirty:
	# 			break

	# 		randomIx = np.random.randint(0, len(iterator))
	# 		key = iterator.reader.cacheKey(randomIx)

	# 	if dirty:
	# 		print("[CachedDatasetReader] Cache is dirty. Rebuilding...")
	# 		buildDirty(baseIterator, iterator, self.cache)

	# @overrides
	# def __getitem__(self, index):
	# 	cacheFile = self.cacheKey(index)
	# 	if self.cache.check(cacheFile):
	# 		item = self.cache.get(cacheFile)
	# 	else:
	# 		item = super().__getitem__(index)
	# 		self.cache.set(cacheFile, item)
	# 	return item

	@overrides
	def __str__(self) -> str:
		summaryStr = "[Cached Dataset Reader]"
		summaryStr += "\n - Cache: %s. Build cache: %s" % (self.cache, self.buildCache)
		summaryStr += "\n %s" % str(self.baseReader)
		return summaryStr
