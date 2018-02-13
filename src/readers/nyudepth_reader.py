import h5py
import numpy as np
from .dataset_reader import DatasetReader
from transforms import Transformer

class NYUDepthReader(DatasetReader):
	def __init__(self, datasetPath, labelsType="depths", imagesShape=(240, 320, 3), labelShape=(55, 74), \
		normalization="normalize", transforms=[], dataSplit=(60, 30, 10)):
		assert len(imagesShape) == 3 and len(labelShape) == 2
		assert labelsType in ("depths", "segmentation"), "Only \"depths\" and \"segmentation\" supported for now. " + \
			"Expected values \"depths\", \"segmentation\", \"both\""
		assert normalization in ("normalize", "standardize")
		self.datasetPath = datasetPath
		self.imagesShape = imagesShape
		self.labelShape = labelShape
		self.normalization = normalization
		self.transforms = transforms
		self.dataSplit = dataSplit
		self.labelsType = labelsType
		self.dataAugmenter = Transformer(transforms, dataShape=imagesShape, labelShape=labelShape, \
			applyOnDataShapeForLabels=True)
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
			# Use the "none" transform just for its resize operation it does at end.
			yield self.dataAugmenter.applyTransform("none", images, labels, interpolationType)
			# Apply each transform
			for augImages, augLabels in self.dataAugmenter.applyTransforms(images, labels, interpolationType):
				yield augImages, augLabels