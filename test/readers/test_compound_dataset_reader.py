import numpy as np
from typing import List, Any, Iterator, Dict, Callable
from neural_wrappers.readers import CompoundDatasetReader, StaticBatchedDatasetReader, CachedDatasetReader, \
	BatchedDatasetReader, MergeBatchedDatasetReader, DatasetItem
from simple_caching import DictMemory

from test_dataset_reader import DummyDataset
from batched_dataset_reader.test_batched_dataset_reader import Reader as BatchedReader

def mergeItems(items:List[DatasetItem]) -> DatasetItem:
	rgbs = np.stack([item["data"]["rgb"] for item in items], axis=0)
	classes = len(items) * [0]
	item = {"data": {"rgb" : rgbs}, "labels" : {"class" : classes}}
	return item

def batchesFn() -> List[int]:
	# batchSizes = [4, 1, 2, 3], so batch[0] has a size of 4, batch[2] a size of 2 etc.
	batchSizes = np.array([4, 1, 2, 3], dtype=np.int32)
	batches = batchIndexFromBatchSizes(batchSizes)
	return batches

class TestCompoundDatasetReader:
	def test_constructor_1(self):
		reader = CompoundDatasetReader(DummyDataset())
		print(reader)
		assert not reader is None
		assert reader.datasetFormat.dataBuckets == {"data" : ["rgb"], "labels" : ["class"]}
		assert list(reader.datasetFormat.dimGetter.keys()) == ["rgb", "class"]
		for v in reader.datasetFormat.dimGetter.values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform.keys()) == ["data", "labels"]
		assert list(reader.datasetFormat.dimTransform["data"].keys()) == ["rgb"]
		for v in reader.datasetFormat.dimTransform["data"].values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform["labels"].keys()) == ["class"]
		for v in reader.datasetFormat.dimTransform["labels"].values():
			assert isinstance(v, Callable)

	def test_getItem_1(self):
		reader = CompoundDatasetReader(DummyDataset())
		item = reader.iterate()[0]
		rgb = item["data"]["rgb"]
		assert np.abs(reader.dataset[0] - rgb).sum() < 1e-5

	def test_getNumData_1(self):
		reader = CompoundDatasetReader(DummyDataset())
		numData = len(reader)
		assert numData == len(reader.dataset)

	def test_iterateOneEpoch_1(self):
		reader = CompoundDatasetReader(DummyDataset())
		generator = reader.iterateOneEpoch()
		for i, item in enumerate(generator):
			rgb = item["data"]["rgb"]
			assert np.abs(rgb - reader.dataset[i]).sum() < 1e-5
		assert i == len(reader.dataset) - 1

	def test_iterateOneEpoch_2(self):
		reader = CompoundDatasetReader(DummyDataset())
		generator = reader.iterateOneEpoch()
		for i, item in enumerate(generator):
			rgb = item["data"]["rgb"]
			assert np.abs(rgb - reader.dataset[i]).sum() < 1e-5
		assert i == len(reader.dataset) - 1

		reader2 = CompoundDatasetReader(reader)
		generator2 = reader2.iterateOneEpoch()
		for i, item in enumerate(generator2):
			rgb = item["data"]["rgb"]
			assert np.abs(rgb - reader2.dataset[i]).sum() < 1e-5
		assert i == len(reader2.dataset) - 1

		reader3 = CompoundDatasetReader(reader2)
		generator3 = reader3.iterateOneEpoch()
		for i, item in enumerate(generator3):
			rgb = item["data"]["rgb"]
			assert np.abs(rgb - reader3.dataset[i]).sum() < 1e-5
		assert i == len(reader3.dataset) - 1

	def test_iterateForever_1(self):
		reader = CompoundDatasetReader(DummyDataset())
		generator = reader.iterateForever()
		for i, item in enumerate(generator):
			rgb = item["data"]["rgb"]
			ix = i % len(reader.dataset)
			assert np.abs(rgb - reader.dataset[ix]).sum() < 1e-5
			if i == len(reader.dataset) * 3:
				break

class TestCompoundDatasetReaderBatched:
	def test_constructor_1(self):
		reader = CompoundDatasetReader(BatchedReader())
		print(reader)
		assert not reader is None
		assert reader.datasetFormat.dataBuckets == {"data" : ["rgb"], "labels" : ["class"]}
		assert list(reader.datasetFormat.dimGetter.keys()) == ["rgb", "class"]
		for v in reader.datasetFormat.dimGetter.values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform.keys()) == ["data", "labels"]
		assert list(reader.datasetFormat.dimTransform["data"].keys()) == ["rgb"]
		for v in reader.datasetFormat.dimTransform["data"].values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform["labels"].keys()) == ["class"]
		for v in reader.datasetFormat.dimTransform["labels"].values():
			assert isinstance(v, Callable)

	def test_getBatchItem_1(self):
		reader = CompoundDatasetReader(BatchedReader())
		# batches = reader.getBatches()
		g = reader.iterate()
		item = g[0]
		rgb = item[0]["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert np.abs(rgb - reader.dataset[0:4]).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = CompoundDatasetReader(BatchedReader())
		g = reader.iterate()
		item = g[0]
		n = len(g)
		for j in range(100):
			batchItem = g[j % n]
			rgb = batchItem[0]["data"]["rgb"]
			index = g.batches[j % n]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_getBatchItem_3(self):
		reader = BatchedReader()
		reader.getBatches = lambda : BatchedDatasetReader.getBatches(reader)
		reader2 = StaticBatchedDatasetReader(reader, batchSize=3)

		try:
			_ = reader.getBatches()
		except Exception:
			pass

		batches = reader2.getBatches()
		generator = reader2.iterateOneEpoch()
		assert batches == generator.batches
		assert len(batches) == len(generator)
		for i in range(len(generator)):
			index = batches[i]
			batchItem = generator[i]
			rgb = batchItem[0]["data"]["rgb"]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = CompoundDatasetReader(BatchedReader())
		g = reader.iterate()
		batches = g.batches
		batchSizes = g.batchLens
		for j in range(len(g)):
			batchItem, B = next(g)
			try:
				assert B == batchSizes[j % len(g)]
			except Exception as e:
				print(str(e))
				breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = batches[j % len(g)]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

			if j == 100:
				break

	def test_iterateForever_2(self):
		reader = CompoundDatasetReader(CompoundDatasetReader(CompoundDatasetReader(BatchedReader())))
		g = reader.iterate()
		batches = g.batches
		batchSizes = g.batchLens
		n = len(batches)
		for j, (batchItem, B) in enumerate(g):
			try:
				assert B == batchSizes[j % n]
			except Exception as e:
				print(str(e))
				breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = batches[j % n]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

			if j == 100:
				break

class TestComboCompounds:
	def test_combo_1(self):
		reader1 = DummyDataset()
		reader2 = CachedDatasetReader(reader1, cache=DictMemory(), buildCache=False)
		reader3 = MergeBatchedDatasetReader(reader2, mergeFn=mergeItems)
		reader4 = StaticBatchedDatasetReader(reader3, 3)

		g2 = reader2.iterate()
		for i in range(len(g2)):
			assert reader2.cache.check(reader2.cacheKey(g2.indexFn(i))) == False

		g4 = reader4.iterate()
		for i in range(len(g4)):
			_ = next(g4)

		for i in range(len(g2)):
			assert reader2.cache.check(reader2.cacheKey(g2.indexFn(i))) == True

	def test_combo_2(self):
		reader1 = DummyDataset()
		reader2 = CachedDatasetReader(reader1, cache=DictMemory(), buildCache=True)
		reader3 = MergeBatchedDatasetReader(reader2, mergeFn=mergeItems)
		reader4 = StaticBatchedDatasetReader(reader3, 3)

		rgbs4 = []
		g4 = reader4.iterate()
		for i in range(len(g4)):
			items, B = next(g4)
			assert B == g4.batchLens[i]
			rgbs4.append(items["data"]["rgb"])
		rgbs4 = np.concatenate(rgbs4)

		rgbs2 = []
		g2 = reader2.iterate()
		for i in range(len(g2)):
			assert reader2.cache.check(reader2.cacheKey(g2.indexFn(i))) == True
			item = next(g2)
			rgbs2.append(item["data"]["rgb"][None])
		rgbs2 = np.concatenate(rgbs2)
		assert np.abs(rgbs2 - rgbs4).sum() <= 1e-5

def main():
	# TestDatasetReader().test_constructor_1()
	TestCompoundDatasetReaderBatched().test_getBatchItem_1()
	# TestComboCompounds().test_combo_2()
	# TestCompoundDatasetReaderBatched().test_iterateForever_2()
if __name__ == "__main__":
	main()