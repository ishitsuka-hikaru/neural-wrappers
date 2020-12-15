import numpy as np
import os
import pytest
from neural_wrappers.readers import MNISTReader
from neural_wrappers.utilities import getGenerators, npCloseEnough

try:
	# This path must be supplied manually in order to pass these tests
	MNIST_READER_PATH = os.environ["MNIST_READER_PATH"]
	pytestmark = pytest.mark.skipif(False, reason="Dataset path not found.")
except Exception:
	pytestmark = pytest.mark.skip("MNIST Dataset path must be set.", allow_module_level=True)

class TestMNISTReader:
	@pytestmark
	def test_mnist_construct_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.dataBuckets == {"data" : ["images"], "labels" : ["labels"]}

	@pytestmark
	def test_mnist_construct_2(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.dataset["train"]["images"].shape == (60000, 28, 28, 1)
		assert reader.dataset["test"]["images"].shape == (10000, 28, 28, 1)

	@pytestmark
	def test_getNumData_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.getNumData("train") == 60000
		assert reader.getNumData("test") == 10000

	@pytestmark
	def test_getNumIterations_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		batches = [1, 2, 7, 10, 23, 100, 444, 9999, 10000]
		trainExpected = [60000, 30000, 8572, 6000, 2609, 600, 136, 7, 6]
		testExpected = [10000, 5000, 1429, 1000, 435, 100, 23, 2, 1]
		for i in range(len(batches)):
			reader.setBatchSize(batches[i])
			assert trainExpected[i] == reader.getNumIterations("train")
			assert testExpected[i] == reader.getNumIterations("test")

	@pytestmark
	def test_iterate_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		MB = np.random.randint(1, 200)
		reader.setBatchSize(MB)
		generator = reader.iterateForever("train")
		assert not generator is None
		items = next(generator)
		assert not items is None

		rgb, labels = items
		assert rgb.shape == (MB, 28, 28, 1)
		assert labels.shape == (MB, 10)

	@pytestmark
	def test_iterate_2(self):
		reader = MNISTReader(MNIST_READER_PATH)
		reader.setBatchSize(30)
		numIters = reader.getNumIterations("train")
		generator = reader.iterateOneEpoch("train")
		for _ in range(numIters):
			_ = next(generator)
		try:
			_ = next(generator)
			assert False
		except StopIteration:
			pass

	# @pytestmark
	# def test_iterate_3(self):
	# 	reader = MNISTReader(MNIST_READER_PATH)
	# 	generator, numIters = getGenerators(reader, 30, keys=["train"])
	# 	firstRgb, firstLabels = next(generator)
	# 	for _ in range(numIters - 1):
	# 		_ = next(generator)
	# 	firstRgbEpoch2, firstLabelsEpoch2 = next(generator)

	# 	assert npCloseEnough(firstRgb, firstRgbEpoch2)
	# 	assert npCloseEnough(firstLabels, firstLabelsEpoch2)

	# @pytestmark
	# def test_normalization_1(self):
	# 	reader = MNISTReader(MNIST_READER_PATH, normalization="none")
	# 	generator, numIters = getGenerators(reader, 30, keys=["train"])
	# 	firstRGBs = next(generator)[0]

	# 	assert firstRGBs.dtype == np.uint8 and firstRGBs.min() == 0 and firstRGBs.max() == 255

	# @pytestmark
	# def test_normalization_2(self):
	# 	reader = MNISTReader(MNIST_READER_PATH, normalization="min_max_0_1")
	# 	generator, numIters = getGenerators(reader, 30, keys=["train"])
	# 	firstRGBs = next(generator)[0]

	# 	assert firstRGBs.dtype == np.float32 and firstRGBs.min() == 0 and firstRGBs.max() == 1

def main():
	pass
	TestMNISTReader().test_mnist_construct_1()
	TestMNISTReader().test_mnist_construct_2()
	TestMNISTReader().test_getNumData_1()
	TestMNISTReader().test_getNumIterations_1()
	TestMNISTReader().test_iterate_1()
	TestMNISTReader().test_iterate_2()
	TestMNISTReader().test_iterate_3()
	TestMNISTReader().test_normalization_1()
	TestMNISTReader().test_normalization_2()

if __name__ == "__main__":
	main()