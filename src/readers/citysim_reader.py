import h5py
import numpy as np
from .dataset_reader import DatasetReader
from transforms import Transformer

class CitySimReader(DatasetReader):
	def __init__(self, datasetPath, imagesShape=(240, 320, 3), labelShape=(55, 74), transforms=[], \
		validationTransforms=None, dataSplit=(60, 30, 10)):
		assert len(imagesShape) == 3 and len(labelShape) == 2
		self.datasetPath = datasetPath
		self.imagesShape = imagesShape
		self.labelShape = labelShape
		self.transforms = transforms
		self.dataSplit = dataSplit
		self.dataAugmenter = Transformer(transforms, dataShape=imagesShape, labelShape=labelShape, \
			applyOnDataShapeForLabels=True)
		# For training, we may not want to use all the transforms when doing validation to increase speed.
		if validationTransforms == None:
			self.validationAugmenter = self.dataAugmenter
		else:
			self.validationAugmenter = Transformer(validationTransforms, dataShape=imagesShape, \
				labelShape=labelShape, applyOnDataShapeForLabels=True)
		self.setup()

	def setup(self):
		# Check validity of the dataSplit (sums to 100 and positive)
		# TODO: maybe move some of this in a base class, since it's going to be common code anyway (getNumIterations
		#  expects self.numData for example, so it's tied code).
		assert len(self.dataSplit) == 3 and self.dataSplit[0] >= 0 and self.dataSplit[1] >= 0 \
			and self.dataSplit[2] >= 0 and self.dataSplit[0] + self.dataSplit[1] + self.dataSplit[2] == 100

		self.dataset = h5py.File(self.datasetPath, "r")
		trainStartIndex = 0
		testStartIndex = self.dataSplit[0] * len(self.dataset["images"]) // 100
		validationStartIndex = testStartIndex + (self.dataSplit[1] * len(self.dataset["images"]) // 100)

		self.indexes = {
			"train" : (trainStartIndex, testStartIndex),
			"test" : (testStartIndex, validationStartIndex),
			"validation" : (validationStartIndex, len(self.dataset["images"]))
		}

		self.numData = {
			"train" : testStartIndex,
			"test" : validationStartIndex - testStartIndex,
			"validation" : len(self.dataset["images"]) - validationStartIndex
		}

		print("[NYUDepth Reader] Setup complete.")

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")

		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = self.indexes[type][0] + i * miniBatchSize
			endIndex = min(self.indexes[type][0] + (i + 1) * miniBatchSize, self.indexes[type][1])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			numData = endIndex - startIndex

			# Data is already normalized in 0-1
			# N N x 240 x 320 x 3
			images = self.dataset["images"][startIndex : endIndex]
			# N x 240 x 320.
			labels = self.dataset["depths"][startIndex : endIndex]
			labels = 1 - labels
			labels[np.where(labels == 1)] = 0

			interpolationType = "bilinear"
			# Apply each transform
			for augImages, augLabels in augmenter.applyTransforms(images, labels, interpolationType):
				yield augImages, augLabels