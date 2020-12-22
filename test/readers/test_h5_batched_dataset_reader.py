import numpy as np
import os
import h5py
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import H5BatchedDatasetReader, DatasetItem, DatasetIndex

def createDatasetIfNotExist():
	tempFileName = "/tmp/dataset.h5"
	if not os.path.exists(tempFileName):
		file = h5py.File(tempFileName, "w")
		file["rgb"] = np.random.randn(10, 3)
		file["class"] = np.random.randint(0, 2, size=(10, ))
		file["batches"] = [4, 1, 2, 3]
		file.flush()
		file.close()
	return tempFileName

class Reader(H5BatchedDatasetReader):
	def __init__(self):
		datasetPath = createDatasetIfNotExist()
		super().__init__(
			datasetPath,
			dataBuckets = {"data" : ["rgb"], "labels" : ["class"]},
			dimTransform = {}
		)

	@overrides
	def getBatches(self) -> List[int]:
		return self.dataset["batches"][()]

	@overrides
	def getNumData(self) -> int:
		return len(self.dataset["rgb"])

class TestH5BatchedDatasetReader:
	def test_constructor_1(self):
		reader = Reader()
		assert not reader is None
		assert not reader.getDataset() is None

	def test_getBatchItem_1(self):
		reader = Reader()
		b = reader.getBatches()
		item = reader.getBatchItem(reader.getBatchIndex(reader.getBatches(), 0))
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert np.abs(rgb - reader.dataset["rgb"][0:4]).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = Reader()
		batches = reader.getBatches()
		n = len(batches)
		for j in range(100):
			index = reader.getBatchIndex(batches, j % n)
			batchItem = reader.getBatchItem(index)
			rgb = batchItem["data"]["rgb"]
			assert np.abs(rgb - reader.dataset["rgb"][index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = Reader()
		batchSizes = reader.getBatches()
		n = len(batchSizes)
		for j, (batchItem, B) in enumerate(reader.iterateForever()):
			try:
				assert B == batchSizes[j % n]
			except Exception:
				breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = reader.getBatchIndex(batchSizes, j % n)
			assert np.abs(rgb - reader.dataset["rgb"][index.start : index.stop]).sum() < 1e-5

			if j == 100:
				break

def main():
	TestH5BatchedDatasetReader().test_constructor_1()

if __name__ == "__main__":
	main()