import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class NYUDepthReader(DatasetReader):
	def __init__(self, datasetPath, labelsType="depths", imageShape=(240, 320, 3), labelShape=(55, 74), \
		normalization="normalize", transforms=[], validationTransforms=None, dataSplit=(60, 30, 10)):
		assert len(imageShape) == 3 and len(labelShape) == 2
		assert labelsType in ("depths", "segmentation"), "Only \"depths\" and \"segmentation\" supported for now. " + \
			"Expected values \"depths\", \"segmentation\", \"both\""
		assert normalization in ("normalize", "standardize")
		self.datasetPath = datasetPath
		self.imageShape = imageShape
		self.labelShape = labelShape
		self.normalization = normalization
		self.transforms = transforms
		self.dataSplit = dataSplit
		self.labelsType = labelsType
		self.dataAugmenter = Transformer(transforms, dataShape=imageShape, labelShape=labelShape)
		# For training, we may not want to use all the transforms when doing validation to increase speed.
		if validationTransforms == None:
			self.validationAugmenter = self.dataAugmenter
		else:
			self.validationAugmenter = Transformer(validationTransforms, dataShape=imageShape, labelShape=labelShape)
		self.setup()

	def __str__(self):
		return "NYUDepth Reader"

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		numAllData = len(self.dataset["images"])
		self.indexes, self.numData = self.computeIndexesSplit(numAllData)
		print("[NYUDepth Reader] Setup complete. Num data: %d. Train: %d, Test: %d, Validation: %d" % \
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

			labelType = "depths" if self.labelsType == "depths" else "labels"
			# N x 3 x 640 x 480 => N x 480 x 640 x 3
			images = np.swapaxes(self.dataset["images"][startIndex : endIndex], 1, 3).astype(np.float32)
			# N x 640 x 480 => N x 480 x 640. Labels can be "depths" or "labels" (for segmentation), not both yet.
			labels = self.dataset[labelType][startIndex : endIndex]
			labels = np.swapaxes(labels, 1, 2)

			if self.normalization == "standardize":
				assert False, "Standardization not yet supported"
				# img_mean = np.array([122.54539034, 104.78338964, 100.02394636])
				# img_std = np.array([73.74140829, 75.4556151, 78.87249644])

				# images -= img_mean
				# images /= img_std
			# Perhaps a better name ?
			elif self.normalization == "normalize":
				images /= 255
				if self.labelsType == "depths":
					labels /= 10

			interpolationType = "nearest" if self.labelsType == "segmentation" else "bilinear"
			# Apply each transform
			for augImages, augLabels in augmenter.applyTransforms(images, labels, interpolationType):
				yield augImages, augLabels