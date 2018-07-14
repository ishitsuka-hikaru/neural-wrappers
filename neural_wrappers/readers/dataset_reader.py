import numpy as np
from prefetch_generator import BackgroundGenerator
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import standardizeData, minMaxNormalizeData, resize_batch
from functools import partial

# Stubs for identity functions, first is used for 1 parameter f(x) = x, second is used for more than one parameter,
#  such as f(x, y, z) = (x, y, z)
def identity(x, **kwargs):
	return x

def identityVar(*args):
	return args

# Stub for making a list, used by various code parts, where the user may provide a single element for a use-case where
#  he'd have to use a 1-element list. This handles that case, so the overall API uses lists, but user provides
#  just an element. If None, just return None.
def makeList(x):
	return None if type(x) == type(None) else x if type(x) == list else [x]

def minMaxNormalizer(data, type, minimums, maximums):
	return minMaxNormalizeData(data, minimums[type], maximums[type])

def standardizer(data, type, means, stds):
	return minMaxNormalizer(data, means[type], stds[type])

# @brief DatasetReader baseclass, that every reader must inherit. Provides basic interface for constructing
#  a dataset reader, with path to the directory/h5py file, data and label dims, dimension transforms, normalizer and
#  augmentation transforms for each dimension. Both data and labels cand be inexistent, by simply providing a None
#  for the dataDims or labelDims variable.
# Pipeline: raw -> dimTransforms -> normalizer -> augTransforms -> resizer
# @param[in] datasetPath The path to the dataset (directory, h5py file etc.)
# @param[in] dataDims A list representing the dimensions of the data ("rgb", "classes", "depth" etc.) or None
# @param[in] labelDims A list representing the dimensions of the label ("depth", "segmentation", "label", etc.) or None
# @param[in] dataDimTransform
# @param[in] labelDimTransform
# @param[in] dataNormalizer
# @param[in] labelNormalizer
# @param[in] dataAugTransform
# @param[in] labelAugTransform
# @param[in] dataShape
# @param[in] labelShape
class DatasetReader:
	def __init__(self, datasetPath, dataDims, labelDims, dataDimTransform={}, labelDimTransform={}, \
		dataNormalizer={}, labelNormalizer={}, dataAugTransform={}, labelAugTransform={}, dataResizer={}, \
		labelResizer={}, dataFinalTransform={}, labelFinalTransform={}):
		self.datasetPath = datasetPath
		self.dataDims = makeList(dataDims)
		self.labelDims = makeList(labelDims)

		# Define the dictionaries that must be updated by each dataset reader.
		self.means, self.stds, self.maximums, self.minimums = {}, {}, {}, {}

		# Pipeline: Raw -> dimTransform -> normalizer -> augTransform -> resize -> finalTransform -> data
		# Dimension transforms are unique to each dimension. If such a transformation is not defined, it is defaulted
		#  to identity.
		dataDimTransform = self.normalizeInputParameters(dataDimTransform, DatasetReader.requireCallableParams)
		self.dataDimTransform = DatasetReader.populateDictByDims(self.dataDims, dataDimTransform, identity)
		# Normalization can also be defined partially, or using built-in strings, such as "min_max_normalizer",
		#  "standardizer" or "none"/"identity"
		dataNormalizer = self.normalizeInputParameters(dataNormalizer, self.normalizerParams)
		self.dataNormalizer = DatasetReader.populateDictByDims(self.dataDims, dataNormalizer, ("none", identity))
		# Augmentation transforms are applied for each dimension after normalization. If nothing is defined, it is
		 # defaulted to identity. Built-in strings are "none"/"indentity" (TODO: crops/rotates etc.)
		dataAugTransform = self.normalizeInputParameters(dataAugTransform, DatasetReader.trasformerParams)
		self.dataAugTransform = DatasetReader.populateDictByDims(self.dataDims, dataAugTransform, identity)
		self.dataTransformer = Transformer(self.dataAugTransform)
		# Resizing is applied for each dimension independently, after performing augmentation. This is needed because
		#  of the 240x320 data vs 50x70 label cropping problem.
		dataResizer = self.normalizeInputParameters(dataResizer, DatasetReader.resizerParams)
		self.dataResizer = DatasetReader.populateDictByDims(self.dataDims, dataResizer, identity)
		# Final data transform is applied to the data after resizing. Such operation can merge all dimensions in one
		#  singular numpy array input, for example, or do any sort of post-processing to each dimension separately.
		dataFinalTransform = self.normalizeInputParameters(dataFinalTransform, DatasetReader.requireCallableParams)
		self.dataFinalTransform = DatasetReader.populateDictByDims(self.dataDims, dataFinalTransform, identity)

		# Same for labels
		labeblDimTransform = self.normalizeInputParameters(labelDimTransform, DatasetReader.requireCallableParams)
		self.labelDimTransform = DatasetReader.populateDictByDims(self.labelDims, labelDimTransform, identity)
		labelNormalizer = self.normalizeInputParameters(labelNormalizer, self.normalizerParams)
		self.labelNormalizer = DatasetReader.populateDictByDims(self.labelDims, labelNormalizer, ("none", identity))
		labelAugTransform = self.normalizeInputParameters(labelAugTransform, DatasetReader.trasformerParams)
		self.labelAugTransform = DatasetReader.populateDictByDims(self.labelDims, labelAugTransform, identity)
		self.labelTransformer = Transformer(self.labelAugTransform)
		labelResizer = self.normalizeInputParameters(labelResizer, DatasetReader.resizerParams)
		self.labelResizer = DatasetReader.populateDictByDims(self.labelDims, labelResizer, identity)
		labelFinalTransform = self.normalizeInputParameters(labelFinalTransform, DatasetReader.requireCallableParams)
		self.labelFinalTransform = DatasetReader.populateDictByDims(self.labelDims, labelFinalTransform, identity)

	# Some transforms (dataDim and dataFinal) must be user provided (or defaulted to identity). This method just checks
	#  that they are callable.
	def requireCallableParams(param):
		assert hasattr(param, "__call__"), "The user provided callback %s must be callable" % (param)
		return param

	# Update the resizer parameters, so in the end they're a Callable function, that resized the input to some deisre
	#  final size. It can be a tuple (H, W, D3...Dn) or a callable (identity, lycon.resize etc.)
	def resizerParams(resize):
		if type(resize) in (list, tuple):
			return partial(resize_batch, dataShape=resize, type="bilinear")
		else:
			assert hasattr(resize, "__call__"), "The user provided resizer %s must be callable" % (resize)
			return resize

	# Update the augmentation parameters, so that in the end they're simply in a (Str, Callable) format for each
	#  dimension. Also accounts for built-in transformations such as "none"/"identity"
	# (TODO: rest when working on TransformerV2)
	# TODO: perhaps move this in the Transformer class when working at it
	def trasformerParams(transform):
		if transform in ("none", "identity"):
			return (transform, identity)
		# User provided transform
		elif type(transform) in (list, tuple):
			assert type(transform[0]) == str and hasattr(transform[1], "__call__"), \
				("The user provided transform \"%s\" must be a tuple of type (Str, Callable) or must one of the " + \
				"default transforms") % transform[0]
			return transform
		else:
			assert False

	# Update the normalization parameters so that in the end they're in the (Str, Callable) format. Also accounts for
	#  built-in normalizations, such as min_max_normalization, standardization or none.
	# Example: "standardizer" => [("standardizer", DatasetReader.standardizer)]
	# Example: {"rgb":"standardizer", "classes":"none"} => 
	#  { "rgb" : ("standardizer", datasetReader.standardizer), classes : ("none", identity) }
	def normalizerParams(self, normalization):
		if normalization == "min_max_normalization":
			normalizer = partial(minMaxNormalizer, minimums=self.minimums, maximums=self.maximums)
			return (normalization, normalizer)
		elif normalization == "standardization":
			normalizer = partial(standardizer, means=self.means, stds=self.stds)
			return (normalization, normalization)
		elif normalization == "none":
			return (normalization, identity)
		else:
			assert type(normalization[0]) == str and hasattr(normalization[1], "__call__"), \
				("The user provided normalization \"%s\" must be a tuple of type (Str, Callable) or must one " + \
					"of \"standardization\", \"min_max_normalization\", \"none\"") % normalization[0]
			return normalization

	# Generic function that transforms an input string/list/dict into the necessary format for populatDictByDims.
	# Str => [Str]
	# [A, B, C] => [A*, B*, C*]
	# {d1:A, d2:B} => {d1:A*, d2:B*}
	# A* is generated by the specificFunc callback, which handles each specific case
	#  (dataNormalizer, dataTransformer, etc.)
	def normalizeInputParameters(self, currentDict, specificFunc):
		if type(currentDict) == str:
			currentDict = makeList(currentDict)

		assert type(currentDict) in (dict, list, tuple)
		newDict = type(currentDict)()
		if type(currentDict) == dict:
			for key in currentDict:
				newDict[key] = specificFunc(currentDict[key])
		elif type(currentDict) in (list, tuple):
			for i in range(len(currentDict)):
				newDict.append(specificFunc(currentDict[i]))
		return newDict

	# @param[in] dims Dimensions across which we wish to populate the dictionary
	# @param[in] currentDict A partial dictionary (or 1 element, if dims is also of len 1,
	# or n-element list if len(dims) == n), that defines the special cases for populating the output dictionary.
	# All other dims are defaulted to defaultValue.
	# @param[in] defaultValue The default value for all the dimensions that are in dims, but aren't in currentDict
	# @param[out] A complete dictionary that has values for any element in the dims input parameter
	# @example populateDictByDims("rgb", RGBToHSV, identity)
	# @example populateDictByDims(["rgb", "depth"], {"rgb" : RGBToHSV, "depth" : })
	def populateDictByDims(dims, currentDict, defaultValue):
		outputDict = {}
		if type(currentDict) != dict:
			currentDict = makeList(currentDict)
			assert len(currentDict) == len(dims), ("When using special list, instead of providing a dictionary" + \
				" with values for each dimension, all the dimensions must be specified. Dims(%d): %s, but given " + \
				"%d item%s: %s") % (ldefaultValueen(dims), dims, len(currentDict), \
				"s" if len(currentDict) != 1 else "", currentDict)
			currentDict = {dims[i] : currentDict[i] for i in range(len(dims))}

		for dim in dims:
			if not dim in currentDict:
				outputDict[dim] = defaultValue
			else:
				outputDict[dim] = currentDict[dim]
				del currentDict[dim]

		assert len(currentDict) == 0, "Wrong keys were specified into currentDict. Dims: %s. Left keys: %s" % \
			(dims, list(currentDict.keys()))
		return outputDict

class ClassificationDatasetReader(DatasetReader):
	# Classification problems are split into N classes which varies from data to data.
	def getNumberOfClasses(self):
		raise NotImplementedError("Should have implemented this")

	def getClasses(self):
		raise NotImplementedError("Should have implemented this")