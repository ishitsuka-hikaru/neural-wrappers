import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import toCategorical

class MNISTReader(DatasetReader):
	def __init__(self, datasetPath, imagesShape=(28, 28), transforms=["none"], testTransforms=None):
		assert len(imagesShape) == 2
		self.datasetPath = datasetPath
		self.imagesShape = imagesShape
		self.transforms = transforms
		self.testTransforms = testTransforms
		self.dataAugmenter = Transformer(transforms, dataShape=imagesShape)
		if testTransforms is None:
			self.testAugmenter = self.dataAugmenter
		else:
			self.testAugmenter = Transformer(testTransforms, dataShape=imagesShape)
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {
			"train" : 60000,
			"test" : 10000
		}
		print("[MNIST Reader] Setup complete")

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test")
		augmenter = self.dataAugmenter if type == "train" else self.testAugmenter

		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)
		for i in range(numIterations):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			numData = endIndex - startIndex

			images = self.dataset["images"][type][startIndex : endIndex]
			labels = toCategorical(self.dataset["labels"][type][startIndex : endIndex], numClasses=10)

			for augImages, _ in augmenter.applyTransforms(images, labels=None):
				yield augImages, labels

	@staticmethod
	def convert(originalDatasetPath):
		from mnist import MNIST
		mnist_data = MNIST(originalDatasetPath)
		trainData, trainLabels = mnist_data.load_training()
		testData, testLabels = mnist_data.load_testing()

		trainData = np.float32(np.array(trainData).reshape((-1, 28, 28)))
		trainLabels = np.array(trainLabels)
		testData = np.float32(np.array(testData).reshape((-1, 28, 28)))
		testLabels = np.array(testLabels)

		trainData = normalizeData(trainData, mean=33.3859647, std=78.65437)
		testData = normalizeData(testData, mean=33.3859647, std=78.65437)

		print(np.mean(trainData), np.std(trainData))
		print(np.mean(testData), np.std(testData))

		file = h5py.File("mnist.h5", "w")
		data_group = file.create_group("images")
		data_group.create_dataset("train", data=trainData)
		data_group.create_dataset("test", data=testData)

		labels_group = file.create_group("labels")
		labels_group.create_dataset("train", data=trainLabels)
		labels_group.create_dataset("test", data=testLabels)