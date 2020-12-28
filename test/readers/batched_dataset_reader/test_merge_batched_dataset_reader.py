import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import DatasetItem, DatasetReader, MergeBatchedDatasetReader, \
	StaticBatchedDatasetReader, RandomBatchedDatasetReader, CachedDatasetReader
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

class Reader(MergeBatchedDatasetReader):
	def __init__(self, baseReader:DatasetReader):
		super().__init__(baseReader, mergeItems)

	@overrides
	def getBatches(self) -> List[int]:
		# batchSizes = [4, 1, 2, 3], so batch[0] has a size of 4, batch[2] a size of 2 etc.
		batchSizes = np.array([4, 1, 2, 3], dtype=np.int32)
		batches = batchIndexFromBatchSizes(batchSizes)
		return batches

class TestMergeBatchedDatasetReader:
	def test_constructor_1(self):
		reader = Reader(DummyDataset())
		assert not reader is None

	def test_constructor_2(self):
		try:
			reader = Reader(DummyBatchedReader())
			assert False
		except Exception:
			pass

	def test_constructor_3(self):
		reader = Reader(CachedDatasetReader(DummyDataset(), cache=None))
		assert not reader is None
		try:
			reader = Reader(CachedDatasetReader(DummyBatchedReader(), cache=None))
			assert False
		except Exception:
			pass

	def test_getBatchItem_1(self):
		reader = Reader(DummyDataset())
		batches = reader.getBatches()
		item = reader[batches[0]]
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert np.abs(rgb - reader.dataset[0:4]).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = Reader(DummyDataset())
		batches = reader.getBatches()
		item = reader[batches[0]]
		n = len(batches)
		for j in range(100):
			index = batches[j % n]
			batchItem = reader[index]
			rgb = batchItem["data"]["rgb"]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_mergeItems_1(self):
		reader = Reader(DummyDataset())
		item1 = reader.baseReader[0]
		item2 = reader.baseReader[1]
		itemMerged = mergeItems([item1, item2])
		rgb1 = item1["data"]["rgb"]
		rgb2 = item2["data"]["rgb"]
		rgbMerged = itemMerged["data"]["rgb"]
		assert np.abs(rgb1 - rgbMerged[0]).sum() < 1e-5
		assert np.abs(rgb2 - rgbMerged[1]).sum() < 1e-5

	def test_iterateOneEpoch_1(self):
		reader = Reader(DummyDataset())
		generator = reader.iterateOneEpoch()
		batches = reader.getBatches()
		for i, (item, B) in enumerate(generator):
			rgb = item["data"]["rgb"]
			assert B == generator.batchLens[i]
			assert B == (batches[i].stop - batches[i].start)
			assert len(rgb) == generator.batchLens[i]
		assert i == len(generator) - 1

	def test_iterateOneEpoch_StaticBatched_1(self):
		reader = StaticBatchedDatasetReader(Reader(DummyDataset()), 4)
		batches = reader.getBatches()
		assert reader.batchLens == [4, 4, 2]
		item = reader[batches[0]]
		rgb = item["data"]["rgb"]
		assert len(rgb) == 4

	# def test_iterateOneEpoch_RandomBatched_1(self):
	# 	reader = RandomBatchedDatasetReader(Reader(DummyDataset()))
	# 	generator = reader.iterateOneEpoch()
	# 	batches = generator.batches
	# 	for i, (item, B) in enumerate(generator):
	# 		rgb = item["data"]["rgb"]
	# 		assert B == generator.batchLens[i]
	# 		assert B == (batches[i].stop - batches[i].start)
	# 		assert len(rgb) == generator.batchLens[i]
	# 	assert i == len(generator) - 1

def main():
	TestMergeBatchedDatasetReader().test_iterateOneEpoch_StaticBatched_1()
	# TestBatchedDatasetReader().test_mergeItems_1()
	# TestBatchedDatasetReader().test_splitItems_1()
	# TestBatchedDatasetReader().test_mergeSplit_1()

if __name__ == "__main__":
	main()