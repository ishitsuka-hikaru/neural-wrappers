import numpy as np
from prefetch_generator import BackgroundGenerator
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import standardizeData, minMaxNormalizeData
from functools import partial

class DatasetReader:
	def __init__(self, datasetPath, dataShape, labelShape, dataDimensions, labelDimensions, \
		transforms=["none"], normalization="standardization"):
		self.datasetPath = datasetPath
		self.dataShape = dataShape
		self.labelShape = labelShape
		self.dataDimensions = dataDimensions
		self.labelDimensions = labelDimensions
		self.transforms = transforms
		self.normalization = normalization

		self.dataAugmenter = Transformer(transforms, dataShape=dataShape, labelShape=labelShape)
		self.validationAugmenter = Transformer(["none"], dataShape=dataShape, labelShape=labelShape)
		self.doNothing = lambda x : x
		self.means, self.stds, self.maximums, self.minimums, self.postDataProcessing = {}, {}, {}, {}, {}

		if normalization == "min_max_normalization":
			self.normalizer = self.minMaxNormalizer
		elif normalization == "standardization":
			self.normalizer = self.standardizer
		elif normalization == "none":
			self.normalizer = lambda data, type: data
		else:
			assert hasattr(normalization[1], "__call__"), ("The user provided normalization \"%s\" must be a " + \
				"tuple of type (Str, Callable) or must one of \"standardization\", \"min_max_normalization\", " + \
				"\"none\"") % normalization[0]
			self.normalization = normalization[0]
			self.normalizer = partial(normalization[1], obj=self)

	# @brief Generic method that looks into a dataset dictionary, and takes each aasked dimension, concatenates it into
	#  one array and returns it back to the caller.
	# @return One list, where each element is one required dimension, extracted from the allData parameter at given
	#  indexes startIndex and endIndex after all processing was done.
	def getData(self, allData, startIndex, endIndex, requiredDimensions):
		dimList = []
		for dim in requiredDimensions:
			data = allData[dim][startIndex : endIndex]
			data = self.postDataProcessing[dim](data)
			data = self.normalizer(data, dim)
			dimList.append(data)
		return dimList

	# @brief Basic min max normalizer, which receives a batches data (MB x shape) and applies the normalization for
	#  each dimension independelty. Requires the class members minimums and maximums to be defined inside the class
	#  for this normalization to work.
	# @param[in] data The data on which the normalization is applied
	# @param[in] type The type (data dimension) for which the field minimums and maximums are searched into
	def minMaxNormalizer(self, data, type):
		data = np.float32(data)
		if self.numDimensions[type] == 1:
			data = minMaxNormalizeData(data, self.minimums[type], self.maximums[type])
		else:
			for i in range(self.numDimensions[type]):
				data[..., i] = minMaxNormalizeData(data[..., i], self.minimums[type][i], self.maximums[type][i])
		return data

	# @brief Basic standardization normalizer, using same convention as minMaxNormalizer.
	# @param[in] data The data on which the normalization is applied
	# @param[in] type The type (data dimension) for which the field means and stsd are searched into
	def standardizer(self, data, type):
		data = np.float32(data)
		if self.numDimensions[type] == 1:
			data = standardizeData(data, self.means[type], self.stds[type])
		else:
			for i in range(self.numDimensions[type]):
				data[..., i] = standardizeData(data[..., i], self.means[type][i], self.stds[type][i])
		return data

	# Handles all the initilization stuff of a specific dataset object.
	def setup(self):
		raise NotImplementedError("Should have implemented this")

	# Generic generator function, which should iterate once the dataset and yield a minibatch subset every time
	def iterate_once(self, type, miniBatchSize):
		raise NotImplementedError("Should have implemented this")

	# Handles all common code that must be done after the setup is complete, such as computing data indexes for
	#  iteration, checks for errors in shapes etc.
	def postSetup(self):
		numDims = len(self.supportedDimensions)
		assert numDims > 0 and len(self.numDimensions) == numDims
		assert (len(self.numData[Type]) == numDims for Type in ("train", "test", "validation"))
		# Validty checks for data dimensions.
		for data in self.dataDimensions: assert data in self.supportedDimensions, "Got %s" % (data)
		for data in self.labelDimensions: assert data in self.supportedDimensions, "Got %s" % (data)

		# If these values were not added manually, use the default values
		for dim in self.supportedDimensions:
			if not dim in self.means:
				self.means[dim] = [0] * self.numDimensions[dim] if self.numDimensions[dim] > 0 else 0

			if not dim in self.stds:
				self.stds[dim] = [1] * self.numDimensions[dim] if self.numDimensions[dim] > 0 else 1

			if not dim in self.maximums:
				self.maximums[dim] = [255] * self.numDimensions[dim] if self.numDimensions[dim] > 0 else 255

			if not dim in self.minimums:
				self.minimums[dim] = [0] * self.numDimensions[dim] if self.numDimensions[dim] > 0 else 0

			if not dim in self.postDataProcessing:
				self.postDataProcessing[dim] = self.doNothing

		assert len(self.means) == numDims and len(self.stds) == numDims and \
			len(self.minimums) == numDims and len(self.maximums) == numDims

		self.dimensionIndexes = {
			"data" : { },
			"label" : { }
		}

		# Check and compiute dimensionIndexes, required for iterating.
		requiredDataDimensions = 0
		for dim in self.dataDimensions:
			endIndex = self.numDimensions[dim]
			self.dimensionIndexes["data"][dim] = (requiredDataDimensions, endIndex)
			requiredDataDimensions += endIndex

		if requiredDataDimensions > 1:
			assert requiredDataDimensions == self.dataShape[-1], \
				"Expected: numDimensions: %s. Got dataShape: %s for: %s dimensions" % \
				(requiredDataDimensions, self.dataShape, self.dataDimensions)

		requiredLabelDimensions = 0
		for dim in self.labelDimensions:
			endIndex = self.numDimensions[dim]
			self.dimensionIndexes["label"][dim] = (requiredLabelDimensions, endIndex)
			requiredLabelDimensions += endIndex

		if requiredLabelDimensions > 1:
			assert requiredLabelDimensions == self.labelShape[-1], \
				"Expected: numDimensions: %s. Got labelShape: %s for: %s dimensions" % \
				(requiredLabelDimensions, self.labelShape, self.labelDimensions)

	# Computes the class members indexes and numData, which represent the amount of data of each portion of the
	#  datset (train/test/val) as well as the starting indexes
	def computeIndexesSplit(self, numAllData):
		# Check validity of the dataSplit (sums to 100 and positive)
		assert len(self.dataSplit) == 3 and self.dataSplit[0] >= 0 and self.dataSplit[1] >= 0 \
			and self.dataSplit[2] >= 0 and self.dataSplit[0] + self.dataSplit[1] + self.dataSplit[2] == 100

		trainStartIndex = 0
		testStartIndex = self.dataSplit[0] * numAllData // 100
		validationStartIndex = testStartIndex + (self.dataSplit[1] * numAllData // 100)

		indexes = {
			"train" : (trainStartIndex, testStartIndex),
			"test" : (testStartIndex, validationStartIndex),
			"validation" : (validationStartIndex, numAllData)
		}

		numSplitData = {
			"train" : testStartIndex,
			"test" : validationStartIndex - testStartIndex,
			"validation" : numAllData - validationStartIndex
		}
		return indexes, numSplitData

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
		assert len(self.transforms), "No transforms used, perhaps set just \"none\""
		return N if accountTransforms == False else N * len(self.transforms)

	def __str__(self):
		return "General dataset reader. Update __str__ in your dataset for more details when using summary."

	def summary(self):
		summaryStr = "[Dataset summary]\n"
		summaryStr += self.__str__() + "\n"

		summaryStr += "Data dimensions: %s. Label dimensions: %s\n" % (self.dataDimensions, self.labelDimensions)
		summaryStr += "Num data: %s\n" % (self.numData)
		summaryStr += "Normalization: %s\n" % (self.normalization)
		summaryStr += "Transforms(%i): %s\n" % (len(self.transforms), self.transforms)
		return summaryStr

class ClassificationDatasetReader(DatasetReader):
	# Classification problems are split into N classes which varies from data to data.
	def getNumberOfClasses(self):
		raise NotImplementedError("Should have implemented this")

	def getClasses(self):
		raise NotImplementedError("Should have implemented this")
