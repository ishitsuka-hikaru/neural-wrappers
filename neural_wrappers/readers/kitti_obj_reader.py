import numpy as np
import h5py
from .dataset_reader import DatasetReader

class KITTIObjReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], normalization="standardization", \
		dataDimensions=["rgb"], labelDimensions=["depth"], baseDataGroup="camera_2", trainValSplit=(80, 20)):

		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, labelDimensions, \
			transforms, normalization)
		self.baseDataGroup = baseDataGroup
		self.dataSplit = (trainValSplit[0], 0, trainValSplit[1])
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.supportedDimensions = ["rgb", "labels", "calibration"]

		self.numDimensions = {
			"rgb" : 3,
			"labels" : 0,
			"calibration" : 0
		}

		print(list(self.dataset["train"]["camera_2"].keys()))
		numAllData = len(self.dataset["train"][self.baseDataGroup]["rgb"])
		indexes, numSplitData = self.computeIndexesSplit(numAllData)
		self.numData = numSplitData
		self.numData["test"] = len(self.dataset["test"][self.baseDataGroup]["rgb"])
		self.indexes = indexes
		self.indexes["test"] = (0, self.numData["test"])

		# TODO: temp hack for multiple label shape in tranformer. I need to think how to allow this case in transformer
		#  as well, but perhaps without adding an additional for (because this is a rare case, where the label is not
		#  a standard shape).
		# Remove the labelShape from Transformer, so we only send the data to the transformer, while ignoring the
		#  label
		self.dataAugmenter.labelShape = None
		self.validationAugmenter.labelShape = None
		self.postSetup()

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		Type = type if type in ("train", "test") else "train"
		thisData = self.dataset[Type][self.baseDataGroup]

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = self.indexes[type][0] + i * miniBatchSize
			endIndex = self.indexes[type][0] + min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			data = self.getData(thisData, startIndex, endIndex, self.dataDimensions)
			data = np.concatenate(data, axis=-1)
			labels = self.getData(thisData, startIndex, endIndex, self.labelDimensions, normalizer=self.doNothing)
			# depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augData, _ in augmenter.applyTransforms(data, None, interpolationType="bilinear"):
				yield augData, labels
				del augData
