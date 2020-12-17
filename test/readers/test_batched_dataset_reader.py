import numpy as np
from overrides import overrides
from typing import Tuple, List
from neural_wrappers.readers import BatchedDatasetReader, DatasetItem, DatasetIndex
from test_dataset_reader import DummyDataset

class Reader(BatchedDatasetReader):
	def __init__(self, baseReader):
		super().__init__(baseReader)

	@overrides
	def getBatchSizes(self) -> List[int]:
		return self.getNumData() * [1]

	# merge(i1, b1, i2, b2) -> i(1,2)
	@overrides
	def mergeItems(self, item1:DatasetItem, batch1:int, item2:DatasetItem, batch2:int) -> DatasetItem:
		rgb1, rgb2 = item1["data"]["rgb"], item2["data"]["rgb"]
		rgbStacked = np.concatenate([rgb1, rgb2], axis=0)
		return {"data" : {"rgb" : rgbStacked}, "labels" : {"class" : [0, 0]}}

	# split(i(1,2), sz1, sz2) -> i1, i2
	@overrides
	def splitItems(self, item:DatasetItem, size1:int, size2:int) -> Tuple[DatasetItem, DatasetItem]:
		pass

	@overrides
	def getItem(self, i:DatasetIndex) -> Tuple[DatasetItem, int]:
		item, b = super().getItem(i)
		rgb = item["data"]["rgb"]
		return {"data" : {"rgb" : np.expand_dims(rgb, axis=0)}, "labels" : {"class" : [0]}}, b

class TestDatasetReader:
	def test_constructor_1(self):
		reader = Reader(DummyDataset())
		item, B = reader.getItem(0)
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 1
		assert B == 1
		assert np.abs(rgb[0] - reader.baseReader.dataset[0]).sum() < 1e-5

	def test_mergeItems_1(self):
		reader = Reader(DummyDataset())
		item1, B1 = reader.getItem(0)
		item2, B2 = reader.getItem(1)
		itemMerged = reader.mergeItems(item1, B1, item2, B2)
		rgb1 = item1["data"]["rgb"]
		rgb2 = item2["data"]["rgb"]
		rgbMerged = itemMerged["data"]["rgb"]
		assert np.abs(rgb1 - rgbMerged[0]).sum() < 1e-5
		assert np.abs(rgb2 - rgbMerged[1]).sum() < 1e-5

def main():
	TestDatasetReader().test_constructor_1()
	TestDatasetReader().test_mergeItems_1()

if __name__ == "__main__":
	main()