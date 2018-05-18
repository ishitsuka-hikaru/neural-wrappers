import numpy as np
import h5py
from .dataset_reader import DatasetReader

# Structure:
# "train"
# 	"raw"
#  		"rgb"
# 		"depth"
#		...
#   "raw_10"
#  		"rgb"
# 		"depth"
# 	"standard" (NOT YET)
#		"rgb"
#		"labels"
# "validation"
# ...

# KITTI Reader class, used with the data already converted in h5py format.
# @param[in] datasetPath Path the the cityscapes_v2.h5 file
# @param[in] imageShape The shape of the images. Must coincide with what type of data is required.
# @param[in] labelShape The shape of the labels (depths).
# @param[in] transforms A list of transformations (augmentations) that are applied to both images and labels
# @param[in] normalization Type of normalization to be applied to the data. Values: "none", "standarization",
#  "min_max_normalization"
# @param[in] dataDimensions A list of all type of inputs that are to be generated by this reader. Supported values
#  are: "rgb"
# @param[in] labelsDimensions A list of all types of labels that are to be generated by this reader. Supported values
#  are: "depth"
class KITTIReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], normalization="standardization", \
		dataDimensions=["rgb"], labelDimensions=["depth"], baseDataGroup="raw_10"):
		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, labelDimensions, \
			transforms, normalization)
		assert baseDataGroup in ("raw", "raw_10")
		self.baseDataGroup = baseDataGroup
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.supportedDimensions = ["rgb", "depth"]

		# Validty checks for data dimensions.
		for data in self.dataDimensions:
			assert data in ("rgb", ), "Got %s" % (data)
		for label in self.labelDimensions:
			assert label in ("depth", ), "Got %s" % (data)

		self.numData = {Type : len(self.dataset[Type][self.baseDataGroup]["rgb"]) for Type in ("train", "validation")}

		# These values are directly computed on the training set of the sequential data (superset of original dataset).
		# They are duplicated for sequential and non-sequential data to avoid unnecessary code.
		self.means = {
			"rgb" : [95.26087859651416, 100.74530927690631, 95.87131461394335],
			"depth" : 645.7766624941177
		}

		self.stds = {
			"rgb" : [79.82532566402132, 81.79278558397813, 83.15537019246743],
			"depth" : 1838.126104311719
		}

		self.maximums = {
			"rgb" : [255, 255, 255],
			"depth" : 65535
		}

		self.minimums = {
			"rgb" : [0, 0, 0],
			"depth" : 0
		}

		self.numDimensions = {
			"rgb" : 3,
			"depth": 1
		}

		self.postSetup()
		print(("[KITTI Reader] Setup complete. Num data: Train: %d, Validation: %d. " + \
			"Images shape: %s. Depths shape: %s. Normalization type: %s") % \
			(self.numData["train"], self.numData["validation"], self.dataShape, self.labelShape, self.normalization))

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		thisData = self.dataset[type][self.baseDataGroup]

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			images = self.getData(thisData, startIndex, endIndex, self.dataDimensions)
			images = np.concatenate(images, axis=-1)
			depths = self.getData(thisData, startIndex, endIndex, self.labelDimensions)
			depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths