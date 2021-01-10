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
		file.create_dataset("batches", (4, ), dtype=h5py.special_dtype(vlen=np.dtype("int32")))
		file["batches"][:] = [[0, 1, 2, 3], [4], [5, 6], [7, 8, 9]]
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
	def __len__(self) -> int:
		return len(self.dataset["rgb"])

class TestH5BatchedDatasetReader:
	def test_constructor_1(self):
		reader = Reader()
		assert not reader is None
		assert not reader.getDataset() is None

	def test_getBatchItem_1(self):
		reader = Reader()
		g = reader.iterate()
		b = g.batches
		index = b[0]
		item = g[0][0]
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert np.abs(rgb - reader.dataset["rgb"][0:4]).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = Reader()
		g = reader.iterate()
		n = len(g)
		for j in range(100):
			index = g.batches[j % n]
			batchItem = g[j % n]
			rgb = batchItem[0]["data"]["rgb"]
			print(index)
			Range = np.arange(index[0], index[-1] + 1)
			assert np.abs(rgb - reader.dataset["rgb"][Range]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = Reader()
		batches = reader.getBatches()
		n = len(batches)
		g = reader.iterateForever()
		for j, (batchItem, B) in enumerate(g):
			try:
				assert B == g.batchLens[j % n]
			except Exception:
				breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = batches[j % n]
			assert np.abs(rgb - reader.dataset["rgb"][index]).sum() < 1e-5

			if j == 100:
				break

def main():
	TestH5BatchedDatasetReader().test_iterateForever_1()

if __name__ == "__main__":
	main()