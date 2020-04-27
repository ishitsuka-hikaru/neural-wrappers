import h5py
import numpy as np
from .classification_dataset_reader import ClassificationDatasetReader
from neural_wrappers.utilities import toCategorical

class Cifar10Reader(ClassificationDatasetReader):
	def __init__(self, datasetPath, dataDims=["images"], labelDims=["labels"], \
		dimTransform={"images" : np.float32, "labels" : lambda x : toCategorical(x, numClasses=10)}, \
		normalizer={}, augTransform=[], resizer={}):
		super().__init__(datasetPath, allDims=["images", "labels"], dataDims=dataDims, labelDims=labelDims, \
			dimTransform=dimTransform, normalizer=normalizer, augTransform=augTransform, resizer=resizer)
		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {
			"train" : 50000,
			"test" : 10000
		}

		self.means = {
			"images" : [125.306918046875, 122.950394140625, 113.86538318359375]
		}

		self.stds = {
			"images" : [62.993219278136884, 62.08870764001421, 66.70489964063091]
		}

		self.minimums = {
			"images" : np.array([0, 0, 0])
		}

		self.maximums = {
			"images" : np.array([255, 255, 255])
		}

		print("[MNIST Reader] Setup complete")

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test")

		dataset = self.dataset[type]
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)

		for i in range(numIterations):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			numData = endIndex - startIndex

			for items in self.getData(dataset, startIndex, endIndex):
				data, labels = items
				yield items

	def getNumberOfClasses(self):
		return 10

	def getClasses(self):
		return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]