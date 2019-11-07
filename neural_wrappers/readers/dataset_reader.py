import numpy as np
from prefetch_generator import BackgroundGenerator
from neural_wrappers.utilities import standardizeData, minMaxNormalizeData, \
	resize_batch, identity, makeList, isSubsetOf
from functools import partial
from collections import OrderedDict

def minMaxNormalizer(data, dim, obj):
	min = obj.minimums[dim]
	max = obj.maximums[dim]
	return minMaxNormalizeData(data, min, max)

def standardizer(data, dim, obj):
	mean = obj.means[dim]
	std = obj.stds[dim]
	return standardizeData(data, mean, std)

# @brief DatasetReader baseclass, that every reader must inherit. Provides basic interface for constructing
#  a dataset reader, with path to the directory/h5py file, data and label dims, dimension transforms, and normalizer
#  for each dimension. Both data and labels cand be inexistent, by simply providing a None for the dataDims
#  or labelDims variable.
# Pipeline: dimGetter (raw) -> dimTransforms -> normalizer -> resizer -> finalTransforms (data and labels)
# @param[in] datasetPath The path to the dataset (directory, h5py file etc.)
# @param[in] dataDims A list representing the dimensions of the data ("rgb", "classes", "depth" etc.) or None
# @param[in] labelDims A list representing the dimensions of the label ("depth", "segmentation", "label", etc.) or None
class DatasetReader:
	def __init__(self, datasetPath, allDims, dataDims, labelDims, dimGetter={}, dimTransform={}, normalizer={}, \
		resizer={}, dataFinalTransform={}, labelFinalTransform={}):

		# Define the dictionaries that must be updated by each dataset reader.
		self.datasetPath = datasetPath
		self.means, self.stds, self.maximums, self.minimums = {}, {}, {}, {}
		self.numData = {"train" : 0, "validation" : 0, "test" : 0}
		self.dataDims = makeList(dataDims)
		self.labelDims = makeList(labelDims)

		# Sanity check
		allDims = makeList(allDims)
		assert isSubsetOf(self.dataDims, allDims) and isSubsetOf(self.labelDims, allDims), ("Exepcted dataDims " + \
			"(%s) and labelDims (%s) to be a subset of allDims (%s)") % (self.dataDims, self.labelDims, allDims)
		# Small efficiency trick, as we only care about the dims in dataDims and labelDims, so no need to perofrm the
		#  pipeline for other unused ones, just to drop them at the very end.
		self.allDims = set(list(self.dataDims) + list(self.labelDims))
		# print(self.allDims, set(self.allDims))
		# assert len(self.allDims) == len(set(self.allDims))
		# Also, if in any level of processing a dimension is given, that was not specified in dataDims or labelDims,
		#  remove it, as it is unused.
		for dim in allDims:
			if dim in self.allDims:
				continue
			for Dict in [dimTransform, normalizer, resizer]:
				if dim in Dict:
					del Dict[dim]

		# Pipeline: dimGetter -> dimTransform -> normalizer -> resize -> finalTransform -> data
		# This pipe-line is applied for both dataDims and labelDims simultaneously, but they are separated at the very
		#  end before providing the data to the user.

		# Dimension getter are used to acquire data for each dimension. Traditionally an indexable dataset is used,
		#  which, using the start/end indexes, the current dimension and the dictoanry, we get the data. However,
		#  more complicated use-cases can be found, such as generating the data in real time.
		#  TODO: see if i can use functools/partial to stop using the dim twice (lambda and dict)
		dimGetter = self.normalizeInputParameters(dimGetter, DatasetReader.requireCallableParams)
		dimGetterDefault = lambda dataset, dim, startIndex, endIndex : dataset[dim][startIndex : endIndex]
		self.dimGetter = DatasetReader.populateDictByDims(self.allDims, dimGetter, dimGetterDefault)
		# Dimension transforms are unique to each dimension. If such a transformation is not defined, it is defaulted
		#  to identity.
		dimTransform = self.normalizeInputParameters(dimTransform, DatasetReader.requireCallableParams)
		self.dimTransform = DatasetReader.populateDictByDims(self.allDims, dimTransform, identity)
		# Normalization can also be defined partially, or using built-in strings, such as "min_max_normalizer",
		#  "standardizer" or "none"/"identity"
		normalizer = self.normalizeInputParameters(normalizer, self.normalizerParams)
		self.normalizer = DatasetReader.populateDictByDims(self.allDims, normalizer, ("identity", identity))
		# Resizing is applied for each dimension independently, after performing augmentation. This is needed because
		#  of the 240x320 data vs 50x70 label cropping problem.
		resizer = self.normalizeInputParameters(resizer, DatasetReader.resizerParams)
		self.resizer = DatasetReader.populateDictByDims(self.allDims, resizer, identity)
		# Final data transform is applied to the data after resizing. Such operation can merge all dimensions in one
		#  singular numpy array input, for example, or do any sort of post-processing to each dimension separately.
		dataFinalTransform = self.normalizeInputParameters(dataFinalTransform, DatasetReader.requireCallableParams)
		self.dataFinalTransform = DatasetReader.populateDictByDims(self.dataDims, dataFinalTransform, identity)
		labelFinalTransform = self.normalizeInputParameters(labelFinalTransform, DatasetReader.requireCallableParams)
		self.labelFinalTransform = DatasetReader.populateDictByDims(self.labelDims, labelFinalTransform, identity)

	### Stuff used for initialization and info about the data that is processed by this reader ###

	# Some transforms (dataDim and dataFinal) must be user provided (or defaulted to identity). This method just checks
	#  that they are callable.
	def requireCallableParams(param):
		assert hasattr(param, "__call__"), "The user provided callback %s must be callable" % (param)
		return param

	# Update the resizer parameters, so in the end they're a Callable function, that resized the input to some deisre
	#  final size. It can be a tuple (H, W, D3...Dn) or a callable (identity, lycon.resize etc.)
	def resizerParams(resize):
		if type(resize) == tuple:
			assert len(resize) >= 2
			return partial(resize_batch, height=resize[0], width=resize[1], interpolation="bilinear")
		else:
			assert hasattr(resize, "__call__"), "The user provided resizer %s must be callable" % (resize)
			return resize

	# Update the normalization parameters so that in the end they're in the (Str, Callable) format. Also accounts for
	#  built-in normalizations, such as min_max_normalization, standardization or none.
	# Example: "standardizer" => [("standardizer", DatasetReader.standardizer)]
	# Example: {"rgb":"standardizer", "classes":"none"} =>
	#  { "rgb" : ("standardizer", datasetReader.standardizer), classes : ("none", identity) }
	def normalizerParams(self, normalization):
		if normalization == "min_max_normalization":
			normalizer = partial(minMaxNormalizer, obj=self)
			return (normalization, normalizer)
		elif normalization == "standardization":
			normalizer = partial(standardizer, obj=self)
			return (normalization, normalizer)
		elif normalization in ("none", "identity"):
			return (normalization, identity)
		else:
			assert type(normalization) == tuple, "Expected a normalization of type (Str, Callable), got %s" \
				% (normalization)
			assert type(normalization[0]) == str and hasattr(normalization[1], "__call__"), \
				("The user provided normalization \"%s\" must be a tuple of type (Str, Callable) or must one " + \
					"of \"standardization\", \"min_max_normalization\", \"none\"") % normalization[0]
			return normalization

	# Generic function that transforms an input (incomplete) dict into the necessary format for populatDictByDims.
	# {d1:A, d2:B} => {d1:A*, d2:B*}
	# A* is generated by the specificFunc callback, which handles each specific case: (normalizer, resizer, etc.).
	def normalizeInputParameters(self, currentDict, specificFunc):
		assert type(currentDict) == dict, "Expected dict, got %s: %s" % (type(currentDict), currentDict)
		newDict = {}
		for key in currentDict:
			newDict[key] = specificFunc(currentDict[key])
		return newDict

	# @param[in] dims Dimensions across which we wish to populate the dictionary
	# @param[in] currentDict A partial dictionary (or 1 element, if dims is also of len 1,
	#  or n-element list if len(dims) == n), that defines the special cases for populating the output dictionary.
	#  All other dims are defaulted to defaultValue.
	# @param[in] defaultValue The default value for all the dimensions that are in dims, but aren't in currentDict
	# @param[out] A complete dictionary that has values for any element in the dims input parameter
	# @example populateDictByDims("rgb", RGBToHSV, identity)
	# @example populateDictByDims(["rgb", "depth"], {"rgb" : RGBToHSV, "depth" : identity})
	def populateDictByDims(dims, currentDict, defaultValue):
		outputDict = {}
		if type(currentDict) != dict:
			currentDict = makeList(currentDict)
			assert len(currentDict) == len(dims), ("When using special list, instead of providing a dictionary" + \
				" with values for each dimension, all the dimensions must be specified. Dims(%d): %s, but given " + \
				"%d item%s: %s") % (len(dims), dims, len(currentDict), \
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

	### Stuff used for iterating through the data provided by this reader ###

	# First 3 steps (acquire data, dimTransform and normalizer) can be applied at once
	def retrieveItem(self, dataset, dim, startIndex, endIndex):
		item = self.dimGetter[dim](dataset, dim, startIndex, endIndex)
		item = self.dimTransform[dim](item)
		item = self.normalizer[dim][1](item, dim=dim)
		return item

	# Pipeline: Raw -> dimTransform -> normalizer -> resize -> finalTransform -> data
	def getData(self, dataset, startIndex, endIndex):
		# First 3 steps (acquire data, dimTransform and normalizer) can be applied at once
		data = {dim : self.retrieveItem(dataset, dim, startIndex, endIndex) for dim in self.allDims}
		finalData = OrderedDict(zip(self.dataDims, len(self.dataDims) * [None]))
		finalLabels = OrderedDict(zip(self.labelDims, len(self.labelDims) * [None]))

		for dim in self.allDims:
			item = data[dim]
			item = self.resizer[dim](item)

			if dim in self.dataDims:
				# Ensure that if we're using the same keys in data and labels, we get copies in each dictionary
				# This is done for safety reasons as we'd probably like to manipulate the data differently
				if dim in self.labelDims:
					finalData[dim] = self.dataFinalTransform[dim](np.copy(item))
				else:
					finalData[dim] = self.dataFinalTransform[dim](item)
			if dim in self.labelDims:
				finalLabels[dim] = self.labelFinalTransform[dim](item)
		return finalData, finalLabels

	# Generic infinite generator, that simply does a while True over the iterate_once method, which only goes one epoch
	# @param[in] type The type of processing that is generated by the generator (typicall train/test/validation)
	# @param[in] miniBatchSize How many items are generated at each step
	# @param[in] maxPrefetch How many items in advance to be generated and stored before they are consumed. If 0, the
	#  thread API is not used at all. If 1, the thread API is used with a queue of length 1 (still works better than
	#  normal in most cases, due to the multi-threaded nature. For length > 1, the queue size is just increased.
	def iterate(self, type, miniBatchSize, maxPrefetch=0):
		assert maxPrefetch >= 0
		while True:
			iterateGenerator = self.iterate_once(type, miniBatchSize)
			if maxPrefetch > 0:
				iterateGenerator = BackgroundGenerator(iterateGenerator, max_prefetch=maxPrefetch)
			for items in iterateGenerator:
				yield items
				del items

	def computeSplitIndexesNumData(numAllData, dataSplit):
		# Check validity of the dataSplit (sums to 100 and positive)
		assert sum([dataSplit[key] for key in dataSplit]) == 100

		currentStartIndex = 0
		numData = {}
		splitIndexes = {}
		for key in dataSplit:
			currentNumItems = numAllData * dataSplit[key] // 100
			currentEndIndex = currentStartIndex + currentNumItems
			numData[key] = currentNumItems
			splitIndexes[key] = (currentStartIndex, currentEndIndex)
			currentStartIndex = currentEndIndex

		return splitIndexes, numData

	# Finds the number of iterations needed for each type, given a miniBatchSize. Eachs transformations adds a new set
	#  of parameters. If none are present then just one set of parameters
	# @param[in] type The type of data from which this is computed (e.g "train", "test", "validation")
	# @param[in] miniBatchSize How many data from all the data is taken at every iteration
	# @param[in] accountTransforms Take into account transformations or not. True value is used in neural_network
	#  wrappers, so if there are 4 transforms, the amount of required iterations for one epoch is numData * 4.
	#  Meanwhile, in reader classes, all transforms are done in the same loop (see NYUDepthReader), so these all
	#  represent same epoch. Defaults to True, so end-users when training networks aren't required to specify it.
	def getNumIterations(self, type, miniBatchSize, accountTransforms=True):
		N = self.numData[type] // miniBatchSize + (self.numData[type] % miniBatchSize != 0)
		return N

	def iterate_once(self, type, miniBatchSize):
		raise NotImplementedError("Should have implemented this")

	def summary(self):
		summaryStr = "[Dataset summary]\n"
		summaryStr += self.__str__() + "\n"

		summaryStr += "All dims: %s. Data dims: %s. Label dims: %s\n" % (self.allDims, self.dataDims, self.labelDims)
		summaryStr += "Dim transforms: %s\n" % (self.dimTransform)
		normalizersStr = {dim : self.normalizer[dim][0] for dim in self.normalizer}
		summaryStr += "Normalizers: %s\n" % (normalizersStr)

		return summaryStr

	# Flatten the indexes array, create a resulting array of shape and type according to resizer.
	# Equivalent of smart-indexing for np.array, but for H5 Dataset
	# Example: getDataFromH5(dataset["train"], "rgb", [[1, 3], [15, 13]], (320, 320, 3))
	# @param[in] dataset The h5 Dataset where we're looking for items
	# @param[in] key The key inside the h5 Dataset
	# @param[in] indexes An array of indexes where we are trying to retrieve data
	def getDataFromH5(self, dataset, key, indexes):
		# Flatten the indexes [[1, 3], [15, 13]] => [1, 3, 15, 13]
		flattenedIndexes = indexes.flatten()

		# Need to get first item, so we can infer the dtype after all transforms are made.
		firstItem = self.retrieveItem(dataset, key, flattenedIndexes[0], flattenedIndexes[0] + 1)
		# firstItem[0] because retrieveItem will give a shape of (1 x actualShape)
		firstItem = self.resizer[key](firstItem)[0]

		resultingShape = [*indexes.shape, *firstItem.shape]
		flattenedResultsShape = [len(flattenedIndexes), *firstItem.shape]
		results = np.zeros(flattenedResultsShape, dtype=firstItem.dtype)
		results[0] = firstItem

		for i in range(1, len(flattenedIndexes)):
			index = flattenedIndexes[i]
			item = self.retrieveItem(dataset, key, index, index + 1)
			item = self.resizer[key](item)[0]
			results[i] = item

		results = results.reshape(resultingShape)
		return results

	def __str__(self):
		return "General dataset reader. Update __str__ in your dataset for more details when using summary."

class ClassificationDatasetReader(DatasetReader):
	# Classification problems are split into N classes which varies from data to data.
	def getNumberOfClasses(self):
		raise NotImplementedError("Should have implemented this")

	def getClasses(self):
		raise NotImplementedError("Should have implemented this")
