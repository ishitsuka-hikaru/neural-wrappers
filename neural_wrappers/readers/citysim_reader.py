import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class CitySimReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], \
		normalization="min_max_normalization", dataDimensions=["rgb"], labelDimensions=["depth"], \
		baseDataGroup="bragadiru_1", dataSplit=(80, 0, 20), **kwargs):
		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, \
			labelDimensions, transforms, normalization)

		self.baseDataGroup = baseDataGroup
		self.dataSplit = dataSplit
		self.kwargs = kwargs
		self.setup()

	def __str__(self):
		return "CitySim Reader"

	def hvnTwoDims(self, images):
		hvn = np.zeros((2, *images.shape), dtype=np.float32)
		whereH = np.where(images == 0)
		whereV = np.where(images == 1)

		hvn[0][whereH] = 1
		hvn[1][whereV] = 1
		hvn = np.transpose(hvn, [1, 2, 3, 0])
		return hvn

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.supportedDimensions = ("rgb", "depth", "hvn")

		numAllData = len(self.dataset[self.baseDataGroup]["rgb"])
		self.indexes, self.numData = self.computeIndexesSplit(numAllData)

		hvnNumDims = 1
		if "hvn" in self.dataDimensions:
			assert "hvnTransform" in self.kwargs
			self.hvnTransform = self.kwargs["hvnTransform"]

			assert self.hvnTransform in ("none", "hvn_two_dims")
			if self.hvnTransform == "none":
				prepareHvn = lambda x : np.expand_dims(x, axis=-1)
			elif self.hvnTransform == "hvn_two_dims":
				hvnNumDims = 2
				prepareHvn = self.hvnTwoDims
			self.postDataProcessing["hvn"] = prepareHvn


		self.minimums = {
			"rgb" : [0, 0, 0],
			"depth" : 0,
			"hvn" : [0] * hvnNumDims if hvnNumDims > 1 else 0
		}

		self.maximums = {
			"rgb" : [255, 255, 255],
			"depth" : 38.837765,
			"hvn" : [1] * hvnNumDims if hvnNumDims > 1 else 1
		}

		self.means = {
			"rgb" : [121.64251106041375, 113.22833753162723, 110.21073242969062],
			"depth" : 11.086505,
			"hvn" : [0] * hvnNumDims if hvnNumDims > 1 else 0
		}

		self.stds = {
			"rgb" : [55.31661791016009, 47.809744429727445, 45.23408344688476],
			"depth" : 5.856089,
			"hvn" : [1] * hvnNumDims if hvnNumDims > 1 else 1
		}

		self.numDimensions = {
			"rgb" : 3,
			"depth" : 1,
			"hvn" : hvnNumDims
		}

		self.postSetup()
		print("[CitySim Reader] Setup complete. Base group: %s. Num data: %d. Train: %d, Test: %d, Validation: %d" % \
			(self.baseDataGroup, numAllData, self.numData["train"], self.numData["test"], self.numData["validation"]))

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		dataset = self.dataset[self.baseDataGroup]

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = self.indexes[type][0] + i * miniBatchSize
			endIndex = self.indexes[type][0] + min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			images = self.getData(dataset, startIndex, endIndex, self.dataDimensions)
			images = np.concatenate(images, axis=-1)
			depths = self.getData(dataset, startIndex, endIndex, self.labelDimensions)
			depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths