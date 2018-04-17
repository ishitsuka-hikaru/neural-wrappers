import numpy as np
import h5py
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import standardizeData

# CityScapes Reader class, used with the data already converted in h5py format.
# @param[in] datasetPath Path the the cityscapes_v2.h5 file
# @param[in] imageShape The shape of the images. Must coincide with what type of data is required.

class CityScapesReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], dataDimensions=["rgb"], \
		sequentialData=False):

		for data in dataDimensions:
			assert data in ("rgb", "depth", "flownet2s", "semantic", "rgb_first_frame"), "Got %s" % (data)
			if sequentialData == True:
				assert not data == "semantic", "Semantic data is not available for sequential dataset"
				assert not data == "rgb_first_frame", "RGB First frame is not available for sequential dataset"
				# Only skipFrames=5 is supported now

		if sequentialData:
			dataDimensions = list(map(lambda x : "seq_%s_5" % (x), dataDimensions))

		self.datasetPath = datasetPath
		self.imageShape = imageShape
		self.labelShape = labelShape
		self.transforms = transforms
		self.dataDimensions = dataDimensions
		self.sequentialData = sequentialData

		self.dataAugmenter = Transformer(transforms, dataShape=imageShape, labelShape=labelShape)
		self.validationAugmenter = Transformer(["none"], dataShape=imageShape, labelShape=labelShape)
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")

		# Only skipFrames=5 is supported now
		if self.sequentialData:
			self.numData = {Type : len(self.dataset[Type]["seq_rgb_5"]) for Type in ("test", "train", "validation")}
		else:
			self.numData = {Type : len(self.dataset[Type]["rgb"]) for Type in ("test", "train", "validation")}

		# These values are directly computed on the training set of the sequential data (superset of original dataset).
		# They are duplicated for sequential and non-sequential data to avoid unnecessary code.
		self.means = {
			"rgb" : [74.96715607296854, 84.3387139353354, 73.62945761147961],
			"rgb_first_frame" : [74.96715607296854, 84.3387139353354, 73.62945761147961],
			"depth" : 8277.619363028218,
			"flownet2s" : [-0.6396361, 5.553444],
			"semantic" : 0,
			"seq_rgb_5" : [74.96715607296854, 84.3387139353354, 73.62945761147961],
			"seq_depth_5" : 8277.619363028218,
			"seq_flownet2s_5" : [-0.6396361, 5.553444]
		}

		self.stds = {
			"rgb" : [49.65527668307159, 50.01892939272212, 49.67332749250472],
			"rgb_first_frame" : [49.65527668307159, 50.01892939272212, 49.67332749250472],
			"depth" : 6569.138224069467,
			"flownet2s" : [32.508713, 15.168872],
			"semantic" : 1,
			"seq_rgb_5" : [49.65527668307159, 50.01892939272212, 49.67332749250472],
			"seq_depth_5" : 6569.138224069467,
			"seq_flownet2s_5" : [32.508713, 15.168872]
		}

		self.numDimensions = {
			"rgb" : 3,
			"depth": 1,
			"flownet2s" : 2,
			"semantic" : 1,
			"rgb_first_frame" : 3,
			"seq_rgb_5" : 3,
			"seq_depth_5" : 1,
			"seq_flownet2s_5" : 2
		}

		requiredDimensions = 0
		for data in self.dataDimensions:
			requiredDimensions += self.numDimensions[data]
		assert requiredDimensions == self.imageShape[-1], "Expected: numDimensions: %s. Got imageShape: %s for: %s" % \
			(requiredDimensions, self.imageShape, self.dataDimensions)

		print(("[CityScapes Images Reader] Setup complete. Num data: Train: %d, Test: %d, Validation: %d. " + \
			"Images shape: %s. Depths shape: %s. Required data: %s. Sequential: %s") % (self.numData["train"], \
			self.numData["test"], self.numData["validation"], self.imageShape, self.labelShape, self.dataDimensions, \
			self.sequentialData))

	def normalizer(self, data, type):
		data = np.float32(data)
		if self.numDimensions[type] == 1:
			return standardizeData(data, mean=self.means[type], std=self.stds[type])
		else:
			for i in range(self.numDimensions[type]):
				data[..., i] = standardizeData(data[..., i], mean=self.means[type][i], std=self.stds[type][i])
			return data

	def prepareSemantic(self, image):
		newImage = np.ones((*image.shape, 1), dtype=np.float32)
		labels = {
			"sky" : np.where(image == 23),
			"buildings" : np.where(image == 11),
			"logo" : np.where(image == 1),
			"road" : np.where(image == 7),
			"sidewalk" : np.where(image == 22)
		}

		for key in labels:
			newImage[labels[key]] = 0

		return newImage

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		thisData = self.dataset[type]
		depthKey = "seq_depth_5" if self.sequentialData else "depth"

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			depths = self.normalizer(thisData[depthKey][startIndex : endIndex], depthKey)
			images = []
			for dim in self.dataDimensions:
				item = self.normalizer(thisData[dim][startIndex : endIndex], dim)
				if dim == "semantic":
					item = self.prepareSemantic(item)
				images.append(item)
			images = np.concatenate(images, axis=3)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths
