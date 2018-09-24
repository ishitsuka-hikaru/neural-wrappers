import numpy as np
import h5py
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class KITTIObjReader(DatasetReader):
	def __init__(self, datasetPath, resizer=(375, 1224, 3), normalization="min_max_normalization", \
		baseDataGroup="camera_2", trainValSplit=(80, 20)):
		# (240, 320, 3) => {"rgb" : (240, 320, 3)}, otherwise use dict
		if type(resizer) == tuple:
			resizer = {"rgb" : resizer}

		super().__init__(datasetPath, \
			allDims = ["rgb", "labels", "calibration"], \
			dataDims=["rgb"], labelDims=["labels", "calibration"], \
			dimTransform = {"rgb" : lambda x : np.float32(x)}, \
			normalizer = {"rgb" : normalization}, \
			resizer=resizer, \
			labelFinalTransform=lambda x : x)

		self.baseDataGroup = baseDataGroup
		self.dataset = h5py.File(self.datasetPath, "r")

		# Compute numData
		# self.numData = {item : len(self.dataset[item][self.baseDataGroup]["rgb"]) for item in self.dataset}
		numAllData = len(self.dataset["train"][self.baseDataGroup]["rgb"])
		trainNum, valNum = trainValSplit
		self.indexes, self.numData = DatasetReader.computeSplitIndexesNumData(numAllData, \
			{"train": trainNum, "validation": valNum})

		self.trainTransformer = self.transformer
		self.valTransformer = Transformer(self.allDims, [])

		self.maximums = {
			"rgb" : np.array([255, 255, 255]),
		}

		self.minimums = {
			"rgb" : np.array([0, 0, 0]),
		}

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		if type == "train":
			self.transformer = self.trainTransformer
		else:
			self.transformer = self.valTransformer

		# both train and validation are on "train" key (different indexes). Test (TODO) is separately.
		dataset = self.dataset["train"][self.baseDataGroup]
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)

		typeStartIndex = self.indexes[type][0]
		for i in range(numIterations):
			startIndex = typeStartIndex + i * miniBatchSize
			endIndex = min(typeStartIndex + (i + 1) * miniBatchSize, typeStartIndex + self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			for items in self.getData(dataset, startIndex, endIndex):
				yield items
