import numpy as np
from overrides import overrides
from typing import Any, Iterator, Dict, Callable
from neural_wrappers.readers import CompoundDatasetReader

from test_dataset_reader import DummyDataset
from batched_dataset_reader.test_batched_dataset_reader import Reader as BatchedReader

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
		item = reader[0]
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
		batches = reader.getBatches()
		item = reader[batches[0]]
		rgb = item["data"]["rgb"]
		assert rgb.shape[0] == 4
		assert np.abs(rgb - reader.dataset[0:4]).sum() < 1e-5

	def test_getBatchItem_2(self):
		reader = CompoundDatasetReader(BatchedReader())
		batches = reader.getBatches()
		item = reader[batches[0]]
		n = len(batches)
		for j in range(100):
			index = batches[j % n]
			batchItem = reader[index]
			rgb = batchItem["data"]["rgb"]
			assert np.abs(rgb - reader.dataset[index.start : index.stop]).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = CompoundDatasetReader(BatchedReader())
		batches = reader.getBatches()
		batchSizes = [(x.stop - x.start) for x in batches]
		n = len(batches)
		for j, (batchItem, B) in enumerate(reader.iterateForever()):
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

def main():
	# TestDatasetReader().test_constructor_1()
	TestCompoundDatasetReaderBatched().test_iterateForever_1()

if __name__ == "__main__":
	main()