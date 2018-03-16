import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class CitySimReader(DatasetReader):
	def __init__(self, datasetPath, imageShape=(240, 320, 3), labelShape=(55, 74), transforms=[], \
		validationTransforms=None, dataSplit=(60, 30, 10)):
		assert len(imageShape) == 3 and len(labelShape) == 2
		self.datasetPath = datasetPath
		self.imageShape = imageShape
		self.labelShape = labelShape
		self.transforms = transforms
		self.dataSplit = dataSplit
		self.dataAugmenter = Transformer(transforms, dataShape=imageShape, labelShape=labelShape, \
			applyOnDataShapeForLabels=True)
		# For training, we may not want to use all the transforms when doing validation to increase speed.
		if validationTransforms == None:
			self.validationAugmenter = self.dataAugmenter
		else:
			self.validationAugmenter = Transformer(validationTransforms, dataShape=imageShape, \
				labelShape=labelShape, applyOnDataShapeForLabels=True)
		self.setup()

	def __str__(self):
		return "CitySim Reader"

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		numAllData = len(self.dataset["images"])
		self.indexes, self.numData = self.computeIndexesSplit(numAllData)
		print("[CitySim Reader] Setup complete. Num data: %d. Train: %d, Test: %d, Validation: %d" % \
			(numAllData, self.numData["train"], self.numData["test"], self.numData["validation"]))

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