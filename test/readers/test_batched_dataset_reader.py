import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import BatchedDatasetReader, DatasetItem, DatasetIndex

class Reader(BatchedDatasetReader):
	def __init__(self):
		super().__init__(
			dataBuckets = {"data" : ["rgb"], "labels" : ["class"]},
			dimGetter = {
				"rgb" : (lambda dataset, index : dataset[index]),
				"class" : (lambda dataset, index : (index.stop - index.start) * [0])
			},
			dimTransform = {}
		)
		self.dataset = np.random.randn(10, 3)
		self.batches = np.array([4, 1, 2, 3], dtype=np.int32)

	@overrides
	def getDataset(self) -> Any:
		return self.dataset

	@overrides
	def getNumData(self) -> int:
		return len(self.dataset)

	@overrides
	def getBatchSizes(self) -> List[int]:
		return self.batches

class TestBatchedDatasetReader:
	def test_constructor_1(self):
		reader = Reader()
		assert not reader is None

	def test_getItem_1(self):
		reader = Reader()
		item, B = reader.getItem(0)
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert B == 4
		assert np.abs(rgb - reader.dataset[0:4]).sum() < 1e-5

	def test_getItem_2(self):
		reader = Reader()
		batchSizes = reader.getBatchSizes()
		n = len(batchSizes)
		for j in range(100):
			batchItem, B = reader.getItem(j % n)
			rgb = batchItem["data"]["rgb"]
			index = reader.getIndex(j % n)
			assert B == batchSizes[j % n]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = Reader()
		batchSizes = reader.getBatchSizes()
		n = len(batchSizes)
		for j, (batchItem, B) in enumerate(reader.iterateForever()):
			try:
				assert B == batchSizes[j % n]
			except Exception:ptest_
				breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = reader.getIndex(j % n)
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

			if j == 100:
				break

	def test_splitItems_1(self):
		reader = Reader(DummyDataset())
		rgbMerged = reader.dataset[0 : 2]
		itemMerged = {"data" : {"rgb" : rgbMerged}, "labels" : {"class" : [0, 0]}}
		item1, item2 = reader.splitItems(itemMerged, 1)
		rgb1 = item1["data"]["rgb"]
		rgb2 = item2["data"]["rgb"]
		assert np.abs(rgb1 - rgbMerged[0]).sum() < 1e-5
		assert np.abs(rgb2 - rgbMerged[1]).sum() < 1e-5

	def test_mergeSplit_1(self):
		reader = Reader(DummyDataset())
		item1, B1 = reader.getItem(0)
		item2, B2 = reader.getItem(1)
		rgb1 = item1["data"]["rgb"]
		rgb2 = item2["data"]["rgb"]

		for i in range(3):
			itemMerged = reader.mergeItems(item1, B1, item2, B2)
			item1_split, item2_split = reader.splitItems(itemMerged, 1)
			rgb1_split = item1_split["data"]["rgb"]
			rgb2_split = item2_split["data"]["rgb"]
			assert np.abs(rgb1 - rgb1_split).sum() < 1e-5
			assert np.abs(rgb2 - rgb2_split).sum() < 1e-5

			item1 = {"data" : {"rgb" : rgb1_split}, "labels" : {"class" : [0]}}
			item2 = {"data" : {"rgb" : rgb2_split}, "labels" : {"class" : [0]}}

def main():
<<<<<<< HEAD
	TestBatchedDatasetReader().test_constructor_1()
=======
	TestDatasetReader().test_constructor_1()
	TestDatasetReader().test_mergeItems_1()
	TestDatasetReader().test_splitItems_1()
	TestDatasetReader().test_mergeSplit_1()
>>>>>>> 034f641e2534ddbb31c385d7d55489a7107a665b

if __name__ == "__main__":
	main()