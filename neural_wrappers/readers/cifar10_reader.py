import h5py
import numpy as np
from .dataset_reader import ClassificationDatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import toCategorical

class Cifar10Reader(ClassificationDatasetReader):
	def __init__(self, datasetPath, imagesShape=(32, 32, 3), transforms=["none"], normalization="standardization"):
		assert len(imagesShape) == 3
		super().__init__(datasetPath, imagesShape, None, transforms, normalization)
		self.dataAugmenter = Transformer(transforms, dataShape=imagesShape)
		self.testAugmenter = Transformer(["none"], dataShape=imagesShape)
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {
			"train" : 50000,
			"test" : 10000
		}

		self.numDimensions = {
			"images" : 3
		}

		self.means = {
			"images" : [125.306918046875, 122.950394140625, 113.86538318359375]
		}

		self.stds = {
			"images" : [62.993219278136884, 62.08870764001421, 66.70489964063091]
		}

		self.minimums = {
			"images" : [0, 0, 0]
		}

		self.maximums = {
			"images" : [255, 255, 255]
		}

		print("[Cifar10 Reader] Setup complete")

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test")
		augmenter = self.dataAugmenter if type == "train" else self.testAugmenter
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
		return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]