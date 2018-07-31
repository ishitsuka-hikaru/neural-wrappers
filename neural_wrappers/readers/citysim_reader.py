import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class CitySimReader(DatasetReader):
	def hvnTwoDimsTransform(images):
		hvn = np.zeros((2, *images.shape), dtype=np.float32)
		whereH = np.where(images == 0)
		whereV = np.where(images == 1)

		hvn[0][whereH] = 1
		hvn[1][whereV] = 1
		hvn = np.transpose(hvn, [1, 2, 3, 0])
		return hvn

	def __init__(self, datasetPath, dataDims, labelDims, augTransform=[], resizer={}, hvnTransform="identity", \
		dataGroup="all"):
		assert hvnTransform in ("identity", "identity_long", "hvn_two_dims")
		assert dataGroup in ("bragadiru", "popesti", "london", "bucharest", "all")
		self.hvnTransform = hvnTransform

		if hvnTransform == "identity":
			hvnDimTransform = lambda x : np.expand_dims(x, axis=-1)
		elif hvnTransform == "identity_long":
			hvnDimTransform = lambda x : np.int64(np.expand_dims(x, axis=-1))
		elif hvnTransform == "hvn_two_dims":
			hvnDimTransform = CitySimReader.hvnTwoDimsTransform

		super().__init__(datasetPath, allDims=["rgb", "depth", "hvn_gt_p1"], dataDims=dataDims, labelDims=labelDims, \
			dimTransform = {
				"rgb" : lambda x : np.float32(x),
				"hvn_gt_p1" : hvnDimTransform,
				"depth" : lambda x : np.expand_dims(x, axis=-1)
			}, \
			normalizer = {
				"rgb" : "min_max_normalization",
				"depth" : "min_max_normalization",
				"hvn_gt_p1" : "identity"
			}, \
			augTransform=augTransform, resizer=resizer)

		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {item : len(self.dataset[item]["rgb"]) for item in self.dataset}

		self.minimums = {
			"rgb" : np.array([0, 0, 0]),
			"depth" : np.array([0])
		}

		depthMaxDataGroup = {
			"bragadiru" : 318.469,
			"popesti" : 312.083,
			"london" : 278.518,
			"bucharest" : 323.485,
			"all" : 323.485
		}

		self.maximums = {
			"rgb" : np.array([255, 255, 255]),
			"depth" : np.array([depthMaxDataGroup[dataGroup]])
		}

		self.trainTransformer = self.transformer
		self.valTransformer = Transformer(self.allDims, [])

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test")
		if type == "train":
			self.transformer = self.trainTransformer
		else:
			self.transformer = self.valTransformer

		dataset = self.dataset[type]
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)

		for i in range(numIterations):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			numData = endIndex - startIndex

			for items in self.getData(dataset, startIndex, endIndex):
				yield items