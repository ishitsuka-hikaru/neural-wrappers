import numpy as np
from neural_wrappers.readers import MNISTReader
from neural_wrappers.utilities import getGenerators, npCloseEnough

# This path must be supplied manually in order to pass these tests
MNIST_READER_PATH = "/home/mihai/Public/Datasets/mnist/mnist.h5"

class TestMNISTReader:
	def test_mnist_construct_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.dataBuckets == {"data" : ["rgb"], "labels" : ["labels"]}

	def test_mnist_construct_2(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.dataset["train"]["images"].shape == (60000, 28, 28, 1)
		assert reader.dataset["test"]["images"].shape == (10000, 28, 28, 1)

	def test_getNumData_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.getNumData("train") == 60000
		assert reader.getNumData("test") == 10000

	def test_getNumIterations_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		batches = [1, 2, 7, 10, 23, 100, 444, 9999, 10000]
		trainExpected = [60000, 30000, 8572, 6000, 2609, 600, 136, 7, 6]
		testExpected = [10000, 5000, 1429, 1000, 435, 100, 23, 2, 1]
		for i in range(len(batches)):
			assert trainExpected[i] == reader.getNumIterations("train", batches[i])
			assert testExpected[i] == reader.getNumIterations("test", batches[i])

	def test_iterate_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		MB = np.random.randint(1, 200)
		generator = reader.iterate("train", MB)
		assert not generator is None
		items = next(generator)
		assert not items is None

		rgb, labels = items
		assert rgb.shape == (MB, 28, 28, 1)
		assert labels.shape == (MB, 10)

	def test_iterate_2(self):
		reader = MNISTReader(MNIST_READER_PATH)
		numIters = reader.getNumIterations("train", 30)
		generator = reader.iterateOneEpoch("train", 30)
		for _ in range(numIters):
			_ = next(generator)
		try:
			_ = next(generator)
			assert False
		except StopIteration:
			pass

	def test_iterate_3(self):
		reader = MNISTReader(MNIST_READER_PATH)
		generator, numIters = getGenerators(reader, 30, keys=["train"])
		firstRgb, firstLabels = next(generator)
		for _ in range(numIters - 1):
			_ = next(generator)
		firstRgbEpoch2, firstLabelsEpoch2 = next(generator)

		assert npCloseEnough(firstRgb, firstRgbEpoch2)
		assert npCloseEnough(firstLabels, firstLabelsEpoch2)

	def test_normalization_1(self):
		reader = MNISTReader(MNIST_READER_PATH, normalization="none")
		generator, numIters = getGenerators(reader, 30, keys=["train"])
		firstRGBs = next(generator)[0]

		assert firstRGBs.dtype == np.uint8 and firstRGBs.min() == 0 and firstRGBs.max() == 255

	def test_normalization_2(self):
		reader = MNISTReader(MNIST_READER_PATH, normalization="min_max_0_1")
		generator, numIters = getGenerators(reader, 30, keys=["train"])
		firstRGBs = next(generator)[0]

		assert firstRGBs.dtype == np.float32 and firstRGBs.min() == 0 and firstRGBs.max() == 1
