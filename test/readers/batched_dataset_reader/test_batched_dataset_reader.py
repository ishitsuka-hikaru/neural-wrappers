import numpy as np
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import BatchedDatasetReader, DatasetItem, DatasetIndex
from neural_wrappers.readers.batched_dataset_reader.utils import getBatchIndex

class Reader(BatchedDatasetReader):
	def __init__(self, N=10):
		super().__init__(
			dataBuckets = {"data" : ["rgb"], "labels" : ["class"]},
			dimGetter = {
				"rgb" : (lambda dataset, index : dataset[index]),
				"class" : (lambda dataset, index : (index.stop - index.start) * [0])
			},
			dimTransform = {}
		)
		self.dataset = np.random.randn(N, 3)

	@overrides
	def getDataset(self) -> Any:
		return self.dataset

	@overrides
	def __len__(self) -> int:
		return len(self.dataset)

	def getBatches(self) -> List[int]:
		return np.array([4, 1, 2, 3], dtype=np.int32)

class TestBatchedDatasetReader:
	def test_constructor_1(self):
		reader = Reader()
		assert not reader is None

	def test_getBatchItem_1(self):
		reader = Reader()
		item = reader[getBatchIndex(reader.getBatches(), 0)]
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert np.abs(rgb - reader.dataset[0:4]).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = Reader()
		batches = reader.getBatches()
		n = len(batches)
		for j in range(100):
			index = getBatchIndex(batches, j % n)
			batchItem = reader[index]
			rgb = batchItem["data"]["rgb"]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = Reader()
		batchSizes = reader.getBatches()
		n = len(batchSizes)
		for j, (batchItem, B) in enumerate(reader.iterateForever()):
			try:
				assert B == batchSizes[j % n]
			except Exception as e:
				print(str(e))
				breakpoint()
			rgb = batchItem["data"]["rgb"]
			index = getBatchIndex(batchSizes, j % n)
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

			if j == 100:
				break

def main():
	TestBatchedDatasetReader().test_iterateForever_1()

if __name__ == "__main__":
	main()