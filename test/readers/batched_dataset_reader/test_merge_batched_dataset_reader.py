import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import DatasetItem, DatasetReader, MergeBatchedDatasetReader, \
	StaticBatchedDatasetReader, RandomBatchedDatasetReader, CachedDatasetReader, CompoundDatasetReader
from neural_wrappers.readers.batched_dataset_reader.utils import batchIndexFromBatchSizes

import sys
sys.path.append("..")
from test_dataset_reader import DummyDataset
from test_batched_dataset_reader import Reader as DummyBatchedReader

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

class TestMergeBatchedDatasetReader:
	def test_constructor_1(self):
		reader = MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn)
		assert not reader is None

	def test_constructor_2(self):
		try:
			reader = MergeBatchedDatasetReader(DummyBatchedReader(), mergeItems, batchesFn)
			assert False
		except Exception:
			pass

	def test_constructor_3(self):
		reader = MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn)
		assert not reader is None
		try:
			reader = MergeBatchedDatasetReader(CachedDatasetReader(DummyBatchedReader(), cache=None), batchesFn)
			assert False
		except Exception:
			pass

	def test_getBatchItem_1(self):
		reader = MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn)
		g = reader.iterate()
		item = g[0][0]
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert np.abs(rgb - reader.dataset[0:4]).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn)
		g = reader.iterate()
		batches = g.batches
		item = g[0]
		n = len(batches)
		for j in range(100):
			index = batches[j % n]
			batchItem = g[j % n][0]
			rgb = batchItem["data"]["rgb"]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_mergeItems_1(self):
		reader = MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn)
		gBase = reader.baseReader.iterate()
		item1 = gBase[0]
		item2 = gBase[1]
		itemMerged = mergeItems([item1, item2])
		rgb1 = item1["data"]["rgb"]
		rgb2 = item2["data"]["rgb"]
		rgbMerged = itemMerged["data"]["rgb"]
		assert np.abs(rgb1 - rgbMerged[0]).sum() < 1e-5
		assert np.abs(rgb2 - rgbMerged[1]).sum() < 1e-5

	def test_iterateOneEpoch_1(self):
		reader = MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn)
		generator = reader.iterateOneEpoch()
		batches = reader.getBatches()
		for i, (item, B) in enumerate(generator):
			rgb = item["data"]["rgb"]
			assert B == generator.batchLens[i]
			assert B == (batches[i].stop - batches[i].start)
			assert len(rgb) == generator.batchLens[i]
		assert i == len(generator) - 1

	def test_iterateOneEpoch_2(self):
		reader = StaticBatchedDatasetReader(MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn), 4)
		reader.baseReader.getBatches = reader.getBatches
		batches = reader.getBatches()
		assert reader.batchLens == [4, 4, 2]
		item = reader.iterate()[0][0]
		rgb = item["data"]["rgb"]
		assert len(rgb) == 4

	def test_iterateOneEpoch_3(self):
		reader = DummyDataset()
		reader2 = CompoundDatasetReader(reader)
		reader3 = MergeBatchedDatasetReader(reader2, mergeItems, batchesFn)

		generator = reader3.iterateOneEpoch()
		batches = reader3.getBatches()
		for i, (item, B) in enumerate(generator):
			rgb = item["data"]["rgb"]
			assert B == generator.batchLens[i]
			assert B == (batches[i].stop - batches[i].start)
			assert len(rgb) == generator.batchLens[i]
		assert i == len(generator) - 1

	def test_iterateOneEpoch_4(self):
		reader = RandomBatchedDatasetReader(MergeBatchedDatasetReader(DummyDataset(), mergeItems, batchesFn))
		generator = reader.iterateOneEpoch()
		batches = generator.batches
		for i, (item, B) in enumerate(generator):
			rgb = item["data"]["rgb"]
			assert B == generator.batchLens[i]
			assert B == (batches[i].stop - batches[i].start)
			assert len(rgb) == generator.batchLens[i]
		assert i == len(generator) - 1

class TestRegression:
	def test_regression_1(self):
		B = 4
		N = 10

		try:
			baseReader = DummyDataset(N=10)
			reader = CompoundDatasetReader(baseReader)
			reader = StaticBatchedDatasetReader(MergeBatchedDatasetReader(reader, mergeFn=mergeItems), B)
		except Exception:
			assert False

		try:
			baseReader = StaticBatchedDatasetReader(MergeBatchedDatasetReader(baseReader, mergeFn=mergeItems), B)
			reader = CompoundDatasetReader(baseReader)
			reader = StaticBatchedDatasetReader(MergeBatchedDatasetReader(reader, mergeFn=mergeItems), B)
			assert False
		except Exception:
			pass

def main():
	TestRegression().test_regression_1()
	# TestMergeBatchedDatasetReader().test_iterateOneEpoch_3()
	# TestBatchedDatasetReader().test_mergeItems_1()
	# TestBatchedDatasetReader().test_splitItems_1()
	# TestBatchedDatasetReader().test_mergeSplit_1()

if __name__ == "__main__":
	main()