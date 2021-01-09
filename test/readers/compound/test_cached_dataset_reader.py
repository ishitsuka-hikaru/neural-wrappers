import numpy as np
import simple_caching
from overrides import overrides
from neural_wrappers.readers import CachedDatasetReader, StaticBatchedDatasetReader, RandomIndexDatasetReader
from neural_wrappers.utilities import deepCheckEqual

import sys
sys.path.append("..")
from batched_dataset_reader.test_batched_dataset_reader import Reader as BaseBatchedReader
from test_dataset_reader import DummyDataset as BaseReader

class BatchedReader(BaseBatchedReader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.isCacheable = True

class Reader(BaseReader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.isCacheable = True

class TestCachedDatasetReader:
	def test_constructor_1(self):
		reader = CachedDatasetReader(Reader(), cache=simple_caching.DictMemory())
		assert not reader is None

	def test_getItem_1(self):
		reader = CachedDatasetReader(Reader(), cache=simple_caching.DictMemory(), buildCache=False)
		index = 0
		g = reader.iterate()
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == False
		item = g[index]
		rgb = item["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == True
		itemCache = g[index]
		rgbCache = itemCache["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_getItem_2(self):
		reader = CachedDatasetReader(Reader(), cache=simple_caching.DictMemory(), buildCache=True)
		index = 0
		g = reader.iterate()
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == True
		item = g[index]
		rgb = item["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == True
		itemCache = g[index]
		rgbCache = itemCache["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_iterateOneEpoch_1(self):
		reader = CachedDatasetReader(Reader(), cache=simple_caching.DictMemory(), buildCache=False)
		generator = reader.iterateOneEpoch()
		rgbs = []
		for i in range(len(generator)):
			assert reader.cache.check(reader.cacheKey(generator.indexFn(i))) == False
			item = next(generator)
			rgb = item["data"]["rgb"]
			rgbs.append(rgb)

		generator = reader.iterateOneEpoch()
		for i in range(len(generator)):
			assert reader.cache.check(reader.cacheKey(generator.indexFn(i))) == True
			item = next(generator)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = CachedDatasetReader(Reader(), cache=simple_caching.DictMemory(), buildCache=False)
		generator = reader.iterateForever()

		rgbs = []
		n = len(generator)
		i = 0
		while True:
			if i == 2 * n:
				break

			if i < n:
				assert reader.cache.check(reader.cacheKey(generator.indexFn(i))) == False
				item = next(generator)
				rgb = item["data"]["rgb"]
				rgbs.append(rgb)
			else:
				assert reader.cache.check(reader.cacheKey(generator.indexFn(i - n))) == True
				item = next(generator)
				rgb = item["data"]["rgb"]
				assert np.abs(rgb - rgbs[i - n]).sum() < 1e-5
			i += 1

	def test_dirty_1(self):
		cache = simple_caching.DictMemory()
		
		baseReader = RandomIndexDatasetReader(Reader(N=100))
		reader1 = CachedDatasetReader(baseReader, cache=cache, buildCache=True)
		reader2 = CachedDatasetReader(baseReader, cache=cache, buildCache=False)
		
		gBase = baseReader.iterateForever()
		g1 = reader1.iterateForever()
		g2 = reader2.iterateForever()

		# First, check consistency (even though our dataset is theoretically not cachable!)
		resBase, res1, res2 = [], [], []
		for i in range(len(gBase)):
			resBase.append(next(gBase))
			res1.append(next(g1))
			res2.append(next(g2))
		assert not deepCheckEqual(resBase, res1)
		assert deepCheckEqual(res1, res2)
		
		# Then, rebuild the cache, it should be dirty, thus a new roll should've been generated
		reader3 = CachedDatasetReader(baseReader, cache=cache, buildCache=True)
		g3 = reader3.iterate()
		res3 = []
		for i in range(len(g3)):
			res3.append(next(g3))
		assert not deepCheckEqual(resBase, res3)
		assert not deepCheckEqual(res1, res3)
		assert not deepCheckEqual(res2, res3)

class TestCachedDatasetReaderBatched:
	def test_constructor_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=simple_caching.DictMemory())
		assert not reader is None

	def test_getBatchItem_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=simple_caching.DictMemory(), buildCache=False)
		index = 0
		g = reader.iterate()
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == False
		item = g[index]
		rgb = item[0]["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == True
		itemCache = g[index]
		rgbCache = itemCache[0]["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = CachedDatasetReader(BatchedReader(), cache=simple_caching.DictMemory(), buildCache=True)
		index = 0
		g = reader.iterate()
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == True
		item = g[index]
		rgb = item[0]["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(g.indexFn(index))) == True
		itemCache = g[index]
		rgbCache = itemCache[0]["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_iterateOneEpoch_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=simple_caching.DictMemory(), buildCache=False)
		generator = reader.iterateOneEpoch()
		rgbs = []
		for i in range(len(generator)):
			assert reader.cache.check(reader.cacheKey(generator.indexFn(i))) == False
			item, B = next(generator)
			rgb = item["data"]["rgb"]
			rgbs.append(rgb)

		generator = reader.iterateOneEpoch()
		for i in range(len(generator)):
			assert reader.cache.check(reader.cacheKey(generator.indexFn(i))) == True
			item, B = next(generator)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

	def test_iterateOneEpoch_2(self):
		baseReader = BatchedReader(N=10)
		reader = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 3), cache=simple_caching.DictMemory(), \
			buildCache=False)
		generator = reader.iterateOneEpoch()
		rgbs = []
		for i in range(len(generator)):
			assert reader.cache.check(reader.cacheKey(generator.indexFn(i))) == False
			item, B = next(generator)
			rgb = item["data"]["rgb"]
			rgbs.append(rgb)

		generator1 = reader.iterateOneEpoch()
		assert len(generator) == len(generator1)
		for i in range(len(generator1)):
			assert reader.cache.check(reader.cacheKey(generator1.indexFn(i))) == True
			item, B = next(generator1)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

		reader2 = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 3), cache=simple_caching.DictMemory(), \
			buildCache=False)
		generator2 = reader2.iterateOneEpoch()
		assert len(generator) == len(generator2)
		for i in range(len(generator2)):
			assert reader.cache.check(reader.cacheKey(generator2.indexFn(i))) == True
			item, B = next(generator2)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

		reader3 = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 4), cache=simple_caching.DictMemory(), \
			buildCache=False)
		generator3 = reader3.iterateOneEpoch()
		assert len(generator) != len(generator3)
		for i in range(len(generator3)):
			assert reader.cache.check(reader.cacheKey(generator3.indexFn(i))) == False
			item, B = next(generator3)
			rgb = item["data"]["rgb"]
			assert len(rgbs[i]) != len(rgb)

		reader4 = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 4), cache=reader3.cache, buildCache=False)
		generator4 = reader4.iterateOneEpoch()
		assert len(generator3) == len(generator4)
		for i in range(len(generator4)):
			assert reader4.cache.check(reader.cacheKey(generator4.indexFn(i))) == True

	def test_iterateForever_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=simple_caching.DictMemory(), buildCache=False)
		generator = reader.iterateForever()
		batches = generator.currentGenerator.batches
		rgbs = []
		n = len(generator)
		i = 0
		while True:
			if i == 2 * n:
				break

			if i < n:
				assert reader.cache.check(reader.cacheKey(batches[i])) == False
				item, B = next(generator)
				rgb = item["data"]["rgb"]
				rgbs.append(rgb)
			else:
				assert reader.cache.check(reader.cacheKey(batches[i - n])) == True
				item, B = next(generator)
				rgb = item["data"]["rgb"]
				assert np.abs(rgb - rgbs[i - n]).sum() < 1e-5
			i += 1

	def test_dirty_1(self):
		cache = simple_caching.DictMemory()
		
		baseReader = RandomIndexDatasetReader(BatchedReader(N=100), seed=42)

		reader1 = CachedDatasetReader(baseReader, cache=cache, buildCache=True)
		reader2 = CachedDatasetReader(baseReader, cache=cache, buildCache=False)
		
		gBase = baseReader.iterateForever()
		g1 = reader1.iterateForever()
		g2 = reader2.iterateForever()

		# First, check consistency (even though our dataset is theoretically not cachable!)
		resBase, res1, res2 = [], [], []
		for i in range(len(gBase)):
			resBase.append(next(gBase))
			res1.append(next(g1))
			res2.append(next(g2))
		assert not deepCheckEqual(resBase, res1)
		assert deepCheckEqual(res1, res2)
		
		# Then, rebuild the cache, it should be dirty, thus a new roll should've been generated
		reader3 = CachedDatasetReader(baseReader, cache=cache, buildCache=True)
		g3 = reader3.iterate()
		res3 = []
		for i in range(len(g3)):
			res3.append(next(g3))
		assert not deepCheckEqual(resBase, res3)
		assert not deepCheckEqual(res1, res3)
		assert not deepCheckEqual(res2, res3)

def main():
	# TestCachedDatasetReaderBatched().test_iterateOneEpoch_2()
	TestCachedDatasetReader().test_getItem_1()
	# TestCachedDatasetReaderBatched().test_dirty_1()

if __name__ == "__main__":
	main()