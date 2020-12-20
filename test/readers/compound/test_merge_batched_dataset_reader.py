import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import DatasetItem, DatasetReader
from neural_wrappers.readers.compound import MergeBatchedDatasetReader
import sys
sys.path.append("..")
from test_dataset_reader import DummyDataset

class Reader(MergeBatchedDatasetReader):
	def __init__(self, baseReader:DatasetReader):
		super().__init__(baseReader)
		self.setBatches(np.array([4, 1, 2, 3], dtype=np.int32))

	@overrides
	def setBatches(self, batches):
		self.batches = batches

	@overrides
	def getBatches(self) -> List[int]:
		return self.batches

	# # merge(i1, b1, i2, b2) -> i(1,2)
	@overrides
	def mergeItems(self, items:DatasetItem, batchSize:int) -> DatasetItem:
		rgbs = np.stack([item["data"]["rgb"] for item in items], axis=0)
		classes = len(items) * [0]
		item = {"data": {"rgb" : rgbs}, "labels" : {"class" : classes}}
		return item

class TestMergeBatchedDatasetReader:
	def test_constructor_1(self):
		reader = Reader(DummyDataset())
		assert not reader is None

	def test_getItem_1(self):
		reader = Reader(DummyDataset())
		item, B = reader.getItem(0)
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert B == 4
		assert np.abs(rgb - reader.baseReader.dataset[0:4]).sum() < 1e-5

	def test_getItem_2(self):
		reader = Reader(DummyDataset())
		batchSizes = reader.getBatches()
		n = len(batchSizes)
		for j in range(100):
			batchItem, B = reader.getItem(j % n)
			rgb = batchItem["data"]["rgb"]
			index = reader.getBatchIndex(batchSizes, j % n)
			start, end = index[0], index[-1] + 1
			assert B == batchSizes[j % n]
			assert np.abs(rgb - reader.baseReader.dataset[start : end]).sum() < 1e-5

	# def test_mergeItems_1(self):
	# 	reader = Reader(DummyDataset())
	# 	item1, B1 = reader.getItem(0)
	# 	item2, B2 = reader.getItem(1)
	# 	itemMerged = reader.mergeItems(item1, B1, item2, B2)
	# 	rgb1 = item1["data"]["rgb"]
	# 	rgb2 = item2["data"]["rgb"]
	# 	rgbMerged = itemMerged["data"]["rgb"]
	# 	assert np.abs(rgb1 - rgbMerged[0]).sum() < 1e-5
	# 	assert np.abs(rgb2 - rgbMerged[1]).sum() < 1e-5

	# def test_splitItems_1(self):
	# 	reader = Reader(DummyDataset())
	# 	rgbMerged = reader.dataset[0 : 2]
	# 	itemMerged = {"data" : {"rgb" : rgbMerged}, "labels" : {"class" : [0, 0]}}
	# 	item1, item2 = reader.splitItems(itemMerged, 1)
	# 	rgb1 = item1["data"]["rgb"]
	# 	rgb2 = item2["data"]["rgb"]
	# 	assert np.abs(rgb1 - rgbMerged[0]).sum() < 1e-5
	# 	assert np.abs(rgb2 - rgbMerged[1]).sum() < 1e-5

	# def test_mergeSplit_1(self):
	# 	reader = Reader(DummyDataset())
	# 	item1, B1 = reader.getItem(0)
	# 	item2, B2 = reader.getItem(1)
	# 	rgb1 = item1["data"]["rgb"]
	# 	rgb2 = item2["data"]["rgb"]

	# 	for i in range(3):
	# 		itemMerged = reader.mergeItems(item1, B1, item2, B2)
	# 		item1_split, item2_split = reader.splitItems(itemMerged, 1)
	# 		rgb1_split = item1_split["data"]["rgb"]
	# 		rgb2_split = item2_split["data"]["rgb"]
	# 		assert np.abs(rgb1 - rgb1_split).sum() < 1e-5
	# 		assert np.abs(rgb2 - rgb2_split).sum() < 1e-5

	# 		item1 = {"data" : {"rgb" : rgb1_split}, "labels" : {"class" : [0]}}
	# 		item2 = {"data" : {"rgb" : rgb2_split}, "labels" : {"class" : [0]}}

def main():
	TestMergeBatchedDatasetReader().test_getItem_1()
	# TestBatchedDatasetReader().test_mergeItems_1()
	# TestBatchedDatasetReader().test_splitItems_1()
	# TestBatchedDatasetReader().test_mergeSplit_1()

if __name__ == "__main__":
	main()