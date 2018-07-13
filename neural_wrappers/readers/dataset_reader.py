import numpy as np
from prefetch_generator import BackgroundGenerator
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import standardizeData, minMaxNormalizeData

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
		dataNormalizer={}, labelNormalizer={}, dataAugTransform={}, labelAugTransform={}, dataShape={}, \
		labelShape={}):
		self.datasetPath = datasetPath
		self.dataDims = makeList(dataDims)
		self.labelDims = makeList(labelDims)

		# Pipeline: Raw -> dimTransform -> normalizer -> augTransform -> resize -> finalTransform -> data
		# Dimension transforms are unique to each dimension. If such a transformation is not defined, it is defaulted
		#  to identity.
		self.dataDimTransform = DatasetReader.populateDictByDims(self.dataDims, dataDimTransform, identity)
		dataNormalizer = DatasetReader.dataNormalizerParams(dataNormalizer)
		self.dataNormalizer = DatasetReader.populateDictByDims(self.dataDims, dataNormalizer, ("none", identity))
		self.dataAugTransform = DatasetReader.populateDictByDims(self.dataDims, dataAugTransform, identity)

		self.labelDimTransform = DatasetReader.populateDictByDims(self.labelDims, labelDimTransform, identity)
		labelNormalizer = DatasetReader.dataNormalizerParams(labelNormalizer)
		self.labelNormalizer = DatasetReader.populateDictByDims(self.labelDims, labelNormalizer, ("none", identity))
		self.labelAugTransform = DatasetReader.populateDictByDims(self.labelDims, labelAugTransform, identity)

	# @brief Basic min max normalizer, which receives a batches data (MB x shape) and applies the normalization for
	#  each dimension independelty. Requires the class members minimums and maximums to be defined inside the class
	#  for this normalization to work.
	# @param[in] data The data on which the normalization is applied
	# @param[in] type The type (data dimension) for which the field minimums and maximums are searched into
	def minMaxNormalizer(data, type):
		for i in range(self.numDimensions[type]):
			data[..., i] = minMaxNormalizeData(data[..., i], self.minimums[type][i], self.maximums[type][i])
		return data

	# @brief Basic standardization normalizer, using same convention as minMaxNormalizer.
	# @param[in] data The data on which the normalization is applied
	# @param[in] type The type (data dimension) for which the field means and stsd are searched into
	def standardizer(data, type):
		for i in range(self.numDimensions[type]):
			data[..., i] = standardizeData(data[..., i], self.means[type][i], self.stds[type][i])
		return data

	# Update the normalization parameters so that in the end they're in the (Str, Callable) format. Also accounts for
	#  built-in normalizations, such as min_max_normalization, standardization or none.
	# Example: "standardizer" => [("standardizer", DatasetReader.standardizer)]
	# Example: {"rgb":"standardizer", "classes":"none"} => 
	#  { "rgb" : ("standardizer", datasetReader.standardizer), classes : ("none", Identity) }
	def dataNormalizerParams(currentDict):
		if type(currentDict) == str:
			currentDict = makeList(currentDict)

		def getActualNormalizationFunc(normalization):
			if normalization == "min_max_normalization":
				return (normalization, DatasetReader.minMaxNormalizer)
			elif normalization == "standardization":
				return (normalization, DatasetReader.standardizer)
			elif normalization == "none":
				return (normalization, identity)
			else:
				assert hasattr(normalization[1], "__call__"), ("The user provided normalization \"%s\" must be a " + \
					"tuple of type (Str, Callable) or must one of \"standardization\", \"min_max_normalization\", " + \
					"\"none\"") % normalization[0]
				return normalization

		assert type(currentDict) in (dict, list, tuple)
		newDict = type(currentDict)()
		if type(currentDict) == dict:
			for key in currentDict:
				newDict[key] = getActualNormalizationFunc(currentDict[key])
		elif type(currentDict) in (list, tuple):
			for i in range(len(currentDict)):
				newDict.append(getActualNormalizationFunc(currentDict[i]))
		return newDict

	# @param[in] dims Dimensions across which we wish to populate the dictionary
	# @param[in] currentDict A partial dictionary (or 1 element, if dims is also of len 1,
	# or n-element list if len(dims) == n), that defines the special cases for populating the output dictionary.
	# All other dims are defaulted to defaultValue.
	# @param[in] defaultValue The default value for all the dimensions that are in dims, but aren't in currentDict
	# @param[out] A complete dictionary that has values for any element in the dims input parameter
	# @example populateDictByDims("rgb", RGBToHSV, Identity)
	# @example populateDictByDims(["rgb", "depth"], {"rgb" : RGBToHSV, "depth" : })
	def populateDictByDims(dims, currentDict, defaultValue):
		outputDict = {}
		if type(currentDict) != dict:
			currentDict = makeList(currentDict)
			assert len(currentDict) == len(dims), ("When using special list, instead of providing a dictionary" + \
				" with values for each dimension, all the dimensions must be specified. Dims(%d): %s, but given " + \
				"%d items") % (len(dims), dims, len(currentDict))
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