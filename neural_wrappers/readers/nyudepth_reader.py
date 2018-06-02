import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class NYUDepthReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], \
		normalization="min_max_normalization", dataDimensions=["images"], labelDimensions=["depths"], \
		dataSplit=(80, 0, 20), version="v2"):
		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, labelDimensions, \
			transforms, normalization)
		assert version in ("v1", "v2")
		self.version = version
		self.dataSplit = dataSplit
		self.setup()

	def __str__(self):
		return "NYUDepth Reader"

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.supportedDimensions = ["images", "depths", "labels"]
		numAllData = len(self.dataset["images"])
		self.indexes, self.numData = self.computeIndexesSplit(numAllData)

		# Top 49 classes for each version of the dataset. The 50th is considered "other" and has value of 0.
		if self.version == "v1":
			topClasses = np.array([1296, 99, 964, 423, 81, 1122, 137, 182, 341, 216, 296, 249, 823, 46, 188, 1027, \
				1362, 1360, 67, 357, 1160, 70, 450, 898, 685, 1084, 113, 942, 19, 827, 990, 1210, 683, 1060, 1314, \
				606, 557, 7, 996, 893, 872, 691, 462, 732, 291, 285, 730, 1225, 106])
		elif self.version == "v2":
			topClasses = np.array([21, 11, 3, 157, 5, 83, 28, 19, 59, 64, 88, 80, 7, 4, 36, 89, 42, 122, 169, 119, \
				143, 141, 85, 17, 172, 15, 123, 135, 26, 45, 331, 158, 144, 24, 124, 136, 55, 2, 238, 8, 1, 312, 94, \
				49, 242, 12, 87, 161, 298])

		self.minimums = {
			"images" : [0, 0, 0],
			"depths" : 0,
			"labels" : 0
		}

		self.maximums = {
			"images" : [255, 255, 255],
			"depths" : 10,
			"labels" : 1
		}

		self.means = {
			"images" : [122.54539034418492, 104.78338963563233, 100.02394636162512],
			"depths" : 2.7963083,
			"labels" : 0
		}

		self.stds = {
			"images" : [73.74140829480243, 75.45561510079736, 78.87249644483357],
			"depths" : 1.3860533,
			"labels" : 1
		}

		self.numDimensions = {
			"images" : 3,
			"depths" : 1,
			"labels" : 1
		}

		self.postDataProcessing = {
			"images" : lambda x : np.swapaxes(x, 1, 3),
			"depths" : lambda x : np.swapaxes(x, 1, 2)
		}

		self.postSetup()
		print("[NYUDepth Reader] Setup complete. Num data: %d. Train: %d, Test: %d, Validation: %d" % \
			(numAllData, self.numData["train"], self.numData["test"], self.numData["validation"]))

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = self.indexes[type][0] + i * miniBatchSize
			endIndex = self.indexes[type][0] + min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			images = self.getData(self.dataset, startIndex, endIndex, self.dataDimensions)
			images = np.concatenate(images, axis=-1)
			depths = self.getData(self.dataset, startIndex, endIndex, self.labelDimensions)
			depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths