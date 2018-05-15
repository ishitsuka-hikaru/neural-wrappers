import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class NYUDepthReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], \
		normalization="min_max_normalization", dataDimensions=["images"], labelDimensions=["depths"], \
		dataSplit=(80, 0, 20)):
		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, labelDimensions, \
			transforms, normalization)
		self.dataSplit = dataSplit
		self.setup()

	def __str__(self):
		return "NYUDepth Reader"

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.supportedDimensions = ["images", "depths"]
		numAllData = len(self.dataset["images"])
		self.indexes, self.numData = self.computeIndexesSplit(numAllData)
		print("[NYUDepth Reader] Setup complete. Num data: %d. Train: %d, Test: %d, Validation: %d" % \
			(numAllData, self.numData["train"], self.numData["test"], self.numData["validation"]))

		self.minimums = {
			"images" : [0, 0, 0],
			"depths" : 0
		}

		self.maximums = {
			"images" : [255, 255, 255],
			"depths" : 10
		}

		self.means = {
			"images" : [122.54539034418492, 104.78338963563233, 100.02394636162512],
			"depths" : 2.7963083
		}

		self.stds = {
			"images" : [73.74140829480243, 75.45561510079736, 78.87249644483357],
			"depths" : 1.3860533
		}

		self.numDimensions = {
			"images" : 3,
			"depths" : 1
		}

		self.postDataProcessing = {
			"images" : lambda x : np.swapaxes(x, 1, 3),
			"depths" : lambda x : np.swapaxes(x, 1, 2)
		}

		self.postSetup()

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			images = self.getData(self.dataset, startIndex, endIndex, self.dataDimensions)
			images = np.concatenate(images, axis=-1)
			depths = self.getData(self.dataset, startIndex, endIndex, self.labelDimensions)
			depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")

		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = self.indexes[type][0] + i * miniBatchSize
			endIndex = min(self.indexes[type][0] + (i + 1) * miniBatchSize, self.indexes[type][1])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			numData = endIndex - startIndex

			images = self.getData(self.dataset, startIndex, endIndex, self.dataDimensions)
			images = np.concatenate(images, axis=-1)
			depths = self.getData(self.dataset, startIndex, endIndex, self.labelDimensions)
			depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths
