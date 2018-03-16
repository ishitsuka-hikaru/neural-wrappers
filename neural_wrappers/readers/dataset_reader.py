import sys
sys.path.append("..")
import numpy as np

class DatasetReader:
	# Handles all the initilization stuff of a specific dataset object.
	def setup(self):
		raise NotImplementedError("Should have implemented this")

	# Generic generator function, which should iterate once the dataset and yield a minibatch subset every time
	def iterate_once(self, type, miniBatchSize=None):
		raise NotImplementedError("Should have implemented this")

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

	# Generic infinite generator, that simply does a while True over the iterate_once method (which only goes one epoch)
	def iterate(self, type, miniBatchSize=None):
		while True:
			for items in self.iterate_once(type, miniBatchSize):
				yield items

	# Finds the number of iterations needed for each type, given a miniBatchSize. Eachs transformations adds a new set
	#  of parameters. If none are present then just one set of parameters 
	# @param[in] type The type of data from which this is computed (e.g "train", "test", "validation")
	# @param[in] miniBatchSize How many data from all the data is taken at every iteration
	# @param[in] accountTransforms Take into account transformations or not. True value is used in neural_network
	#  wrappers, so if there are 4 transforms, the amount of required iterations for one epoch is numData * 5.
	#  Meanwhile, in reader classes, all transforms are done in the same loop (see NYUDepthReader), so these all
	#  represent same epoch.
	def getNumIterations(self, type, miniBatchSize, accountTransforms=False):
		N = self.numData[type] // miniBatchSize + (self.numData[type] % miniBatchSize != 0)
		assert len(self.transforms), "No transforms used, perhaps set just \"none\""
		return N if accountTransforms == False else N * len(self.transforms)

	# Returns the mean of the dataset. Sometimes differnt means are used for different shapes (due to rescaling).
	def getMean(self, shape=None):
		raise NotImplementedError("Should have implemented this")

	def getStd(self, shape=None):
		raise NotImplementedError("Should have implemented this")

	def __str__(self):
		return "General dataset reader. Update __str__ in your dataset for more details when using summary."

	def summary(self):
		summaryStr = "[Dataset summary]\n"
		summaryStr += self.__str__() + "\n"

		summaryStr += "Num data: %s\n" % (self.numData)
		summaryStr += "Transforms(%i): %s\n" % (len(self.transforms), self.transforms)
		return summaryStr

class ClassificationDatasetReader(DatasetReader):
	# Classification problems are split into N classes which varies from data to data.
	def getNumberOfClasses(self):
		raise NotImplementedError("Should have implemented this")

	def getClasses(self):
		raise NotImplementedError("Should have implemented this")
