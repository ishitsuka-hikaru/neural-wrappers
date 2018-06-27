import numpy as np
import h5py
from .dataset_reader import DatasetReader
from neural_wrappers.utilities import minMaxNormalizeData

# Structure:
# "train"
# 	"noseq"
#  		"rgb"
# 		"ground_truth_fine"
# 		"flownet2s"
#		...
# 	"seq_5"
#		"rgb"
#		"flownet2s"
#		...
#	"seq_1"
#	...
# "validation"
# ...

def logit(x):
	res1 = np.log((x + 100) / (1 - x + 1e-6))
	res1[np.where(res1 == -np.inf)] = 0
	res1[np.where(res1 == np.inf)] = 0
	res1[np.where(res1 == np.nan)] = 0
	res2 = minMaxNormalizeData(res1, np.min(res1), np.max(res1))
	return res2

def logitReverse(data, type, obj):
	data = obj.minMaxNormalizer(data, type)
	if type == "depth":
		# First reverse the orders of the depth
		data = 1 - data
		data = logit(data)
	return data

# Used for plotting, so we convert a [0-33] input image to a nice RGB semantic plot.
def cityscapesSemanticCmap(label):
	if label.ndim != 2:
		raise ValueError("Expect 2-D input label")

	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift
	ind >>= 3

	if np.max(label) >= len(colormap):
		raise ValueError("label value too large.")

	return np.uint8(colormap[label])

# CityScapes Reader class, used with the data already converted in h5py format.
# @param[in] datasetPath Path the the cityscapes_v2.h5 file
# @param[in] imageShape The shape of the images. Must coincide with what type of data is required.
# @param[in] labelShape The shape of the labels (depths).
# @param[in] transforms A list of transformations (augmentations) that are applied to both images and labels
# @param[in] dataDimensions A list of all type of inputs that are to be generated by this reader. Supported values
#  are: "rgb", "depth", "ground_truth_fine", "rgb_first_frame"
# @param[optional] semanticTransform The type of transformation to be applied for semantic data. Only valid if
#  "ground_truth_fine" is used in dataDimensions.
class CityScapesReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], normalization="standardization", \
		dataDimensions=["rgb"], labelDimensions=["depth"], baseDataGroup=False, semanticTransform=None, \
		opticalFlowTransform=None):
		if normalization == "min_max_normalization_reverse_logit":
			normalization = (normalization, logitReverse)
		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, \
			labelDimensions, transforms, normalization)
		assert baseDataGroup in ("noseq", "seq_5")
		self.baseDataGroup = baseDataGroup
		self.semanticTransform = semanticTransform
		self.opticalFlowTransform = opticalFlowTransform
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.supportedDimensions = ("rgb", "depth", "flownet2s", "ground_truth_fine", "rgb_first_frame", "deeplabv3", \
			"rgb_prev_frame", "depth_prev_frame_deeplabv3_new_dims")

		semanticNumDims = 1
		if "ground_truth_fine" in self.dataDimensions or "deeplabv3" in self.dataDimensions:
			assert self.semanticTransform in ("default", "foreground-background", "none", "semantic_new_dims")
			if self.semanticTransform == "default":
				prepareSemantic = lambda x : np.expand_dims(x, axis=-1) / 33
			elif self.semanticTransform == "foreground-background":
				prepareSemantic = self.semanticFGBG
			elif self.semanticTransform == "none":
				prepareSemantic = lambda x : np.expand_dims(x, axis=-1)
			elif self.semanticTransform == "semantic_new_dims":
				semanticNumDims = 19
				prepareSemantic = self.semanticNewDims

			self.postDataProcessing["ground_truth_fine"] = prepareSemantic
			self.postDataProcessing["deeplabv3"] = prepareSemantic

		opticalFlowNumDimensions = 2
		flownet2sMean = [-0.6396361, 5.553444]
		flownet2sStd = [32.508713, 15.168872]
		flownet2sMaximum = [278.29926, 225.12384]
		flownet2sMinimum = [-494.61987, -166.98322]
		if "flownet2s" in self.dataDimensions:
			assert self.opticalFlowTransform in ("none", "magnitude")
			if self.opticalFlowTransform == "magnitude":
				self.postDataProcessing["flownet2s"] = \
					lambda x : np.expand_dims(np.hypot(x[..., 0], x[..., 1]), axis=-1)
				opticalFlowNumDimensions = 1
				flownet2sMean = 19.803894
				flownet2sStd = 20.923697
				flownet2sMaximum = 496.0213
				flownet2sMinimum = 0

		# Only skipFrames=5 is supported now
		self.numData = {Type : len(self.dataset[Type][self.baseDataGroup]["rgb"]) \
			for Type in ("test", "train", "validation")}

		# These values are directly computed on the training set of the sequential data (superset of original dataset).
		# They are duplicated for sequential and non-sequential data to avoid unnecessary code.
		self.means = {
			"rgb" : [74.96715607296854, 84.3387139353354, 73.62945761147961],
			"rgb_first_frame" : [74.96715607296854, 84.3387139353354, 73.62945761147961],
			"rgb_prev_frame" : [74.96715607296854, 84.3387139353354, 73.62945761147961],
			"depth" : 8277.619363028218,
			"flownet2s" : flownet2sMean,
			"ground_truth_fine" : [0] * semanticNumDims,
			"deeplabv3" : [0] * semanticNumDims,
			"depth_prev_frame_deeplabv3_new_dims" : 8277.619363028218
		}

		self.stds = {
			"rgb" : [49.65527668307159, 50.01892939272212, 49.67332749250472],
			"rgb_first_frame" : [49.65527668307159, 50.01892939272212, 49.67332749250472],
			"rgb_prev_frame" : [49.65527668307159, 50.01892939272212, 49.67332749250472],
			"depth" : 6569.138224069467,
			"flownet2s" : flownet2sStd,
			"ground_truth_fine" : [1] * semanticNumDims if semanticNumDims > 1 else 1,
			"deeplabv3" : [1] * semanticNumDims if semanticNumDims > 1 else 1,
			"depth_prev_frame_deeplabv3_new_dims" : 6569.138224069467
		}

		self.maximums = {
			"rgb" : [255, 255, 255],
			"rgb_first_frame" : [255, 255, 255],
			"rgb_prev_frame" : [255, 255, 255],
			"depth" : 32257,
			"flownet2s" : flownet2sMaximum,
			"ground_truth_fine" : [1] * semanticNumDims if semanticNumDims > 1 else 1,
			"deeplabv3" : [1] * semanticNumDims if semanticNumDims > 1 else 1,
			"depth_prev_frame_deeplabv3_new_dims" : 32257
		}

		self.minimums = {
			"rgb" : [0, 0, 0],
			"rgb_first_frame" : [0, 0, 0],
			"rgb_prev_frame" : [0, 0, 0],
			"depth" : 0,
			"flownet2s" : flownet2sMinimum,
			"ground_truth_fine" : [0] * semanticNumDims if semanticNumDims > 1 else 0,
			"deeplabv3" : [0] * semanticNumDims if semanticNumDims > 1 else 0,
			"depth_prev_frame_deeplabv3_new_dims" : 0
		}

		self.numDimensions = {
			"rgb" : 3,
			"rgb_first_frame" : 3,
			"rgb_prev_frame" : 3,
			"depth": 1,
			"flownet2s" : opticalFlowNumDimensions,
			"ground_truth_fine" : semanticNumDims,
			"deeplabv3" : semanticNumDims,
			"depth_prev_frame_deeplabv3_new_dims" : 1
		}

		self.postSetup()
		print(("[CityScapes Images Reader] Setup complete. Num data: Train: %d, Test: %d, Validation: %d. " + \
			"Images shape: %s. Depths shape: %s. Required data: %s. Base group: %s. Normalization type: %s. " + \
			"Semantic type: %s. Optical Flow type: %s") % \
			(self.numData["train"], self.numData["test"], self.numData["validation"], self.dataShape, \
			self.labelShape, self.dataDimensions, self.baseDataGroup, self.normalization, self.semanticTransform, \
			self.opticalFlowTransform))

	def semanticFGBG(self, images):
		newImage = np.ones((*images.shape, 1), dtype=np.float32)
		labels = {
			"sky" : np.where(images == 23),
			"buildings" : np.where(images == 11),
			"logo" : np.where(images == 1),
			"road" : np.where(images == 7),
			"sidewalk" : np.where(images == 22),
			"sidewalk2": np.where(images == 8),
			"trees" : np.where(images == 21)
		}

		for key in labels:
			newImage[labels[key]] = 0
		return newImage

	def semanticNewDims(self, images):
		importantIds = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
		newImages = np.zeros((19, *images.shape), dtype=np.float32)
		for i in range(len(importantIds)):
			thisOne = newImages[i]
			whereId = np.where(images == importantIds[i])
			thisOne[whereId] = 1
		newImages = np.transpose(newImages, [1, 2, 3, 0])
		return newImages

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		thisData = self.dataset[type][self.baseDataGroup]

		# Hackish solution for the pre-computed depth that is stored as smaller size. We remove it at each iteration
		#  compute the resize/transforms, and append it back. Solution to fix this is to move resize from trasnformer
		#  to someplace else. Transformer should only apply augmentation anways, not resizes. TODO.
		if "depth_prev_frame_deeplabv3_new_dims" in self.dataDimensions:
			depthPrevIndex = self.dataDimensions.index("depth_prev_frame_deeplabv3_new_dims")
			actualFinalIndex = 0
			for i in range(depthPrevIndex):
				actualFinalIndex += self.numDimensions[self.dataDimensions[i]]
			tmpData = np.zeros((miniBatchSize, 870, 1820, 1), dtype=np.float32)
		else:
			depthPrevIndex = None

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			images = self.getData(thisData, startIndex, endIndex, self.dataDimensions)
			if not depthPrevIndex is None:
				depthPrev = images[depthPrevIndex][..., 0]
				images[depthPrevIndex] = tmpData[0 : endIndex - startIndex]
			images = np.concatenate(images, axis=-1)
			depths = self.getData(thisData, startIndex, endIndex, self.labelDimensions)
			depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				if not depthPrevIndex is None:
					augImages[..., actualFinalIndex] = depthPrev
				yield augImages, augDepths
				del augImages, augDepths
