import h5py
import numpy as np
from .dataset_reader import ClassificationDatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import toCategorical

class MNISTReader(ClassificationDatasetReader):
	def __init__(self, datasetPath, imagesShape=(28, 28), transforms=["none"], normalization="standardization"):
		assert len(imagesShape) == 2
		super().__init__(datasetPath, imagesShape, None, None, None, transforms, normalization)
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {
			"train" : 60000,
			"test" : 10000
		}

		self.numDimensions = {
			"images" : 1
		}

		self.means = {
			"images" : 33.318421449829934
		}

		self.stds = {
			"images" : 78.56748998339798
		}

		self.minimums = {
			"images" : 0
		}

		self.maximums = {
			"images" : 255
		}

		print("[MNIST Reader] Setup complete")

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		data = self.dataset[type]
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)

		for i in range(numIterations):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			numData = endIndex - startIndex

			images = self.normalizer(data=data["images"][startIndex : endIndex], type="images")
			labels = toCategorical(data["labels"][startIndex : endIndex], numClasses=10)

			for augImages, _ in augmenter.applyTransforms(images, labels=None):
				yield augImages, labels

	def getNumberOfClasses(self):
		return 10

	def getClasses(self):
		return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]