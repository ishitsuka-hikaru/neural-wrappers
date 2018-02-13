import numpy as np
from .dataset_reader import DatasetReader

class NYUDepthReader(DatasetReader):
	def __init__(self, datasetPath, labels="depths", imagesShape=(240, 320, 3), depthShape=(55, 74), \
		normalization="normalize", transforms=[], dataSplit=(60, 30, 10)):
		assert len(imagesShape) == 3 and len(depthShape) == 2
		assert labels == "depths", "Only depths supported for now. Expected values 'depths', 'labels', 'both'"
		assert normalization in ("normalize", "standardize")
		self.datasetPath = datasetPath
		self.imagesShape = imagesShape
		self.depthShape = depthShape
		self.normalization = normalization
		self.transforms = transforms
		self.dataSplit = dataSplit
		self.dataAugmenter = Transformer(transforms, dataShape=imagesShape, labelShape=depthShape, \
			applyOnDataShapeForLabels=True)
		self.setup()

	def setup(self):
		# Check validity of the dataSplit (sums to 100 and positive)
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

			# N x 3 x 640 x 480 => N x 480 x 640 x 3
			images = np.swapaxes(self.dataset["images"][startIndex : endIndex], 1, 3).astype(np.float32)
			# N x 640 x 480 => N x 480 x 640
			depths = np.swapaxes(self.dataset["depths"][startIndex : endIndex], 1, 2)

			if self.normalization == "standardize":
				img_mean = np.array([122.54539034, 104.78338964, 100.02394636])
				img_std = np.array([73.74140829, 75.4556151, 78.87249644])

				images -= img_mean
				images /= img_std
			# Perhaps a better name ?
			elif self.normalization == "normalize":
				images /= 255
				depths /= 10

			# Use the "none" transform just for its resize operation it does at end.
			yield self.dataAugmenter.applyTransform("none", images, depths)
			# Apply each transform
			for augImages, augDepths in self.dataAugmenter.applyTransforms(images, depths):
				yield augImages, augDepths