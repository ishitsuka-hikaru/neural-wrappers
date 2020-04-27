import numpy as np
from neural_wrappers.readers import MNISTReader

# This path must be supplied manually in order to pass these tests
MNIST_READER_PATH = "/home/mihai/Public/Datasets/mnist/mnist.h5"

class TestMNISTReader:
	def test_mnist_construct_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.allDims == {"data" : ["rgb"], "labels" : ["labels"]}

	def test_get_num_data_1(self):
		reader = MNISTReader(MNIST_READER_PATH)
		assert reader.getNumData("train") == 60000 and reader.getNumData("test") == 10000

	# def test_mnist_construct_12self):
		# reader = MNISTReader(MNIST_READER_PATH)
# 		generator = reader.iterate_once("train", miniBatchSize=5)
# 		assert reader.dataset["images"]["train"].shape == (60000, 28, 28)
# 		assert reader.dataset["images"]["test"].shape == (10000, 28, 28)

# 	def test_mnist_generator_1(self):
# 		reader = MNISTReader(MNIST_READER_PATH)
# 		generator = reader.iterate_once("train", miniBatchSize=5)
# 		assert not generator is None

# 	def test_mnist_generator_2(self):
# 		reader = MNISTReader(MNIST_READER_PATH)
# 		generator = reader.iterate_once("train", miniBatchSize=5)
# 		assert not generator is None

# 		items, labels = next(generator)
# 		assert items.shape == (5, 28, 28)
# 		assert labels.shape == (5, 10)

# 	def test_mnist_generator_2(self):
# 		reader = MNISTReader(MNIST_READER_PATH, imagesShape=(32, 32))
# 		generator = reader.iterate_once("train", miniBatchSize=5)
# 		assert not generator is None

# 		items, labels = next(generator)
# 		assert items.shape == (5, 32, 32)
# 		assert labels.shape == (5, 10)

# 	def test_mnist_generator_3(self):
# 		reader = MNISTReader(MNIST_READER_PATH, imagesShape=(32, 32), transforms=["none", "mirror"])
# 		generator = reader.iterate_once("train", miniBatchSize=5)
# 		assert not generator is None

# 		items1, labels1 = next(generator)
# 		assert items1.shape == (5, 32, 32)
# 		assert labels1.shape == (5, 10)

# 		items2, labels2 = next(generator)
# 		assert items2.shape == (5, 32, 32)
# 		assert labels2.shape == (5, 10)

# 		a = np.mean(items1)
# 		b = np.mean(items2)

# 		assert np.abs(np.mean(items1) - np.mean(items2)) < 1e-8
# 		assert np.abs(np.std(items1) - np.std(items2)) < 1e-8
