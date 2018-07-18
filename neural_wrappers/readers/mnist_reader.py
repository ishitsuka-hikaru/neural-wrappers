import h5py
import numpy as np
from .dataset_reader import ClassificationDatasetReader
from neural_wrappers.utilities import toCategorical
from neural_wrappers.transforms import Transformer

class MNISTReader(ClassificationDatasetReader):
	def __init__(self, datasetPath, dataDims=["images"], labelDims=["labels"], \
		dimTransform={"images" : np.float32, "labels" : lambda x : toCategorical(x, numClasses=10)}, \
		normalizer={}, augTransform=[], resizer={}):
		super().__init__(datasetPath, allDims=["images", "labels"], dataDims=dataDims, labelDims=labelDims, \
			dimTransform=dimTransform, normalizer=normalizer, augTransform=augTransform, resizer=resizer)
		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {
			"train" : 60000,
			"test" : 10000
		}

		self.means = {
			"images" : np.array([33.318421449829934])
		}

		self.stds = {
			"images" : np.array([78.56748998339798])
		}

		self.minimums = {
			"images" : np.array([0])
		}

		self.maximums = {
			"images" : np.array([255])
		}

		self.trainTransformer = self.transformer
		self.valTransformer = Transformer(self.allDims, [])

		print("[MNIST Reader] Setup complete")

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test")
		if type == "train":
			self.transformer = self.trainTransformer
		else:
			self.transformer = self.valTransformer

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
		return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]