import numpy as np
from neural_wrappers.readers import PercentDatasetReader, StaticBatchedDatasetReader

import sys
sys.path.append("..")
from batched_dataset_reader.test_batched_dataset_reader import Reader as BaseBatchedReader
from test_dataset_reader import DummyDataset as BaseReader

class TestPercentDatasetReader:
	def test_constructor_1(self):
		reader = PercentDatasetReader(BaseReader(), percent=100)
		assert not reader is None

	def test_constructor_2(self):
		reader = PercentDatasetReader(BaseReader(), percent=100)
		assert not reader is None

	def test_len_1(self):
		reader = PercentDatasetReader(BaseReader(), percent=100)
		assert len(reader) == len(BaseReader())

	def test_len_2(self):
		reader = PercentDatasetReader(BaseReader(), percent=10)
		assert len(reader) == len(BaseReader()) // 10
		assert len(reader) == 1

	def test_iterateOneEpoch_1(self):
		reader = BaseReader()
		readerHalf = PercentDatasetReader(reader, percent=50)
		assert len(readerHalf) == len(reader) // 2

		generator = reader.iterateOneEpoch()
		generatorHalf = readerHalf.iterateOneEpoch()
		assert len(generatorHalf) == len(generator) // 2
		n = len(generator)
		for i in range(n):
			rgb = next(generator)["data"]["rgb"]
			if i < n // 2:
				rgbHalf = next(generatorHalf)["data"]["rgb"]
				assert np.abs(rgb - rgbHalf).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = BaseReader()
		readerHalf = PercentDatasetReader(reader, percent=50)

		generator = reader.iterateForever()
		generatorHalf = readerHalf.iterateForever()

		n = len(generator)
		rgbs = []
		for i in range(2 * n):
			rgb = next(generator)["data"]["rgb"]
			rgbHalf = next(generatorHalf)["data"]["rgb"]
			if i < n:
				rgbs.append(rgb)
			else:
				assert np.abs(rgb - rgbs[i % n]).sum() < 1e-5

			if i >= n // 2:
				assert np.abs(rgbHalf - rgbs[i % (n//2)]).sum() < 1e-5

class TestPercentDatasetReaderBatched:
	def test_constructor_1(self):
		reader = PercentDatasetReader(BaseBatchedReader(), percent=10)
		assert not reader is None

	def test_constructor_2(self):
		reader = PercentDatasetReader(BaseBatchedReader(), percent=10)
		assert not reader is None

	def test_len_1(self):
		reader = PercentDatasetReader(BaseBatchedReader(), percent=100)
		assert len(reader) == len(BaseReader())

	def test_len_2(self):
		reader = PercentDatasetReader(BaseBatchedReader(), percent=10)
		assert len(reader) == len(BaseReader()) // 10
		assert len(reader) == 1

	def test_iterateOneEpoch_1(self):
		reader = StaticBatchedDatasetReader(BaseBatchedReader(N=100), 10)
		readerHalf = PercentDatasetReader(reader, percent=50)
		assert len(readerHalf) == len(reader) // 2

		generator = reader.iterateOneEpoch()
		generatorHalf = readerHalf.iterateOneEpoch()
		assert len(generatorHalf) == len(generator) // 2

		n = len(generator)
		for i in range(n):
			rgb = next(generator)[0]["data"]["rgb"]
			if i < n // 2:
				rgbHalf = next(generatorHalf)[0]["data"]["rgb"]
				assert np.abs(rgb - rgbHalf).sum() < 1e-5

	def test_iterateForever_1(self):
		reader = StaticBatchedDatasetReader(BaseBatchedReader(N=100), 10)
		readerHalf = PercentDatasetReader(reader, percent=50)

		generator = reader.iterateForever()
		generatorHalf = readerHalf.iterateForever()

		n = len(generator)
		rgbs = []
		for i in range(2 * n):
			rgb = next(generator)[0]["data"]["rgb"]
			rgbHalf = next(generatorHalf)[0]["data"]["rgb"]
			if i < n:
				rgbs.append(rgb)
			else:
				assert np.abs(rgb - rgbs[i % n]).sum() < 1e-5

			if i >= n // 2:
				assert np.abs(rgbHalf - rgbs[i % (n//2)]).sum() < 1e-5

if __name__ == "__main__":
	TestPercentDatasetReader().test_iterateOneEpoch_1()