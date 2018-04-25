import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import toCategorical

class MNISTReader(DatasetReader):
	def __init__(self, datasetPath, imagesShape=(28, 28), transforms=["none"], normalizationType="standardize"):
		assert len(imagesShape) == 2
		super().__init__(datasetPath, imagesShape, None, transforms, normalizationType)
		self.dataAugmenter = Transformer(transforms, dataShape=imagesShape)
		self.testAugmenter = Transformer(["none"], dataShape=imagesShape)
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
		augmenter = self.dataAugmenter if type == "train" else self.testAugmenter
		data = self.dataset[type]
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)

		for i in range(numIterations):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			numData = endIndex - startIndex

			images = self.normalizer(data["images"][startIndex : endIndex], type="images")
			labels = toCategorical(data["labels"][startIndex : endIndex], numClasses=10)

			for augImages, _ in augmenter.applyTransforms(images, labels=None):
				yield augImages, labels
