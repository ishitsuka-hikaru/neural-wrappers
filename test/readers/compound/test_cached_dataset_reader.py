import numpy as np
import pycache
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import CachedDatasetReader, DatasetItem, DatasetIndex, \
	StaticBatchedDatasetReader, getBatchIndex
from neural_wrappers.utilities import getGenerators

import sys
sys.path.append("..")
from test_batched_dataset_reader import Reader as BaseBatchedReader
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
		reader = CachedDatasetReader(Reader(), cache=pycache.DictMemory())
		assert not reader is None

	def test_getItem_1(self):
		reader = CachedDatasetReader(Reader(), cache=pycache.DictMemory())
		index = 0
		assert reader.cache.check(reader.cacheKey(index)) == False
		item = reader[index]
		rgb = item["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(index)) == True
		itemCache = reader[index]
		rgbCache = itemCache["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_getItem_2(self):
		reader = CachedDatasetReader(Reader(), cache=pycache.DictMemory(), buildCache=True)
		index = 0
		assert reader.cache.check(reader.cacheKey(index)) == True
		item = reader[index]
		rgb = item["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(index)) == True
		itemCache = reader[index]
		rgbCache = itemCache["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_iterateOneEpoch_1(self):
		reader = CachedDatasetReader(Reader(), cache=pycache.DictMemory())
		generator = reader.iterateOneEpoch()
		rgbs = []
		for i in range(len(generator)):
			assert reader.cache.check(reader.cacheKey(i)) == False
			item = next(generator)
			rgb = item["data"]["rgb"]
			rgbs.append(rgb)

		generator = reader.iterateOneEpoch()
		for i in range(len(generator)):
			assert reader.cache.check(reader.cacheKey(i)) == True
			item = next(generator)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = CachedDatasetReader(Reader(), cache=pycache.DictMemory())
		generator = reader.iterateForever()

		rgbs = []
		n = len(generator)
		i = 0
		while True:
			if i == 2 * n:
				break

			if i < n:
				assert reader.cache.check(reader.cacheKey(i)) == False
				item = next(generator)
				rgb = item["data"]["rgb"]
				rgbs.append(rgb)
			else:
				assert reader.cache.check(reader.cacheKey(i - n)) == True
				item = next(generator)
				rgb = item["data"]["rgb"]
				assert np.abs(rgb - rgbs[i - n]).sum() < 1e-5
			i += 1

class TestCachedDatasetReaderBatched:
	def test_constructor_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=pycache.DictMemory())
		assert not reader is None

	def test_getBatchItem_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=pycache.DictMemory())
		batches = reader.getBatches()
		index = getBatchIndex(batches, 0)
		assert reader.cache.check(reader.cacheKey(index)) == False
		item = reader[index]
		rgb = item["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(index)) == True
		itemCache = reader[index]
		rgbCache = itemCache["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = CachedDatasetReader(BatchedReader(), cache=pycache.DictMemory(), buildCache=True)
		batches = reader.getBatches()
		index = getBatchIndex(batches, 0)
		assert reader.cache.check(reader.cacheKey(index)) == True
		item = reader[index]
		rgb = item["data"]["rgb"]
		assert reader.cache.check(reader.cacheKey(index)) == True
		itemCache = reader[index]
		rgbCache = itemCache["data"]["rgb"]
		assert np.abs(rgb - rgbCache).sum() < 1e-5

	def test_iterateOneEpoch_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=pycache.DictMemory())
		generator = reader.iterateOneEpoch()
		rgbs = []
		for i in range(len(generator)):
			batchIndex = getBatchIndex(generator.batches, i)
			assert reader.cache.check(reader.cacheKey(batchIndex)) == False
			item, B = next(generator)
			rgb = item["data"]["rgb"]
			rgbs.append(rgb)

		generator = reader.iterateOneEpoch()
		for i in range(len(generator)):
			batchIndex = getBatchIndex(generator.batches, i)
			assert reader.cache.check(reader.cacheKey(batchIndex)) == True
			item, B = next(generator)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

	def test_iterateOneEpoch_2(self):
		baseReader = BatchedReader(N=10)
		reader = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 3), cache=pycache.DictMemory())
		generator = reader.iterateOneEpoch()
		rgbs = []
		for i in range(len(generator)):
			batchIndex = getBatchIndex(generator.batches, i)
			assert reader.cache.check(reader.cacheKey(batchIndex)) == False
			item, B = next(generator)
			rgb = item["data"]["rgb"]
			rgbs.append(rgb)

		generator1 = reader.iterateOneEpoch()
		assert len(generator) == len(generator1)
		for i in range(len(generator1)):
			batchIndex = getBatchIndex(generator1.batches, i)
			assert reader.cache.check(reader.cacheKey(batchIndex)) == True
			item, B = next(generator1)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

		reader2 = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 3), cache=pycache.DictMemory())
		generator2 = reader2.iterateOneEpoch()
		assert len(generator) == len(generator2)
		for i in range(len(generator2)):
			batchIndex = getBatchIndex(generator2.batches, i)
			assert reader.cache.check(reader.cacheKey(batchIndex)) == True
			item, B = next(generator2)
			rgb = item["data"]["rgb"]
			assert np.abs(rgbs[i] - rgb).sum() < 1e-5

		reader3 = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 4), cache=pycache.DictMemory())
		generator3 = reader3.iterateOneEpoch()
		assert len(generator) != len(generator3)
		for i in range(len(generator3)):
			batchIndex = getBatchIndex(generator3.batches, i)
			assert reader.cache.check(reader.cacheKey(batchIndex)) == False
			item, B = next(generator3)
			rgb = item["data"]["rgb"]
			assert len(rgbs[i]) != len(rgb)

		reader4 = CachedDatasetReader(StaticBatchedDatasetReader(baseReader, 4), cache=reader3.cache)
		generator4 = reader4.iterateOneEpoch()
		assert len(generator3) == len(generator4)
		for i in range(len(generator4)):
			batchIndex = getBatchIndex(generator4.batches, i)
			assert reader4.cache.check(reader.cacheKey(batchIndex)) == True

	def test_iterateForever_1(self):
		reader = CachedDatasetReader(BatchedReader(), cache=pycache.DictMemory())
		generator = reader.iterateForever()
		batches = generator.currentGenerator.batches
		rgbs = []
		n = len(generator)
		i = 0
		while True:
			if i == 2 * n:
				break

			if i < n:
				assert reader.cache.check(reader.cacheKey(getBatchIndex(batches, i))) == False
				item, B = next(generator)
				rgb = item["data"]["rgb"]
				rgbs.append(rgb)
			else:
				assert reader.cache.check(reader.cacheKey(getBatchIndex(batches, i - n))) == True
				item, B = next(generator)
				rgb = item["data"]["rgb"]
				assert np.abs(rgb - rgbs[i - n]).sum() < 1e-5
			i += 1

def main():
	TestCachedDatasetReader().test_getBatchItem_2()

if __name__ == "__main__":
	main()