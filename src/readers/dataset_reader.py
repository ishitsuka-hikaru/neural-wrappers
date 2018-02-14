import numpy as np
from utilities.public_api import normalizeData

class DatasetReader:
	# Handles all the initilization stuff of a specific dataset object.
	def setup(self):
		raise NotImplementedError("Should have implemented this")

	# Generic generator function, which should iterate once the dataset and yield a minibatch subset every time
	def iterate_once(self, type, miniBatchSize=None):
		raise NotImplementedError("Should have implemented this")

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

class ClassificationDatasetReader(DatasetReader):
	# Classification problems are split into N classes which varies from data to data.
	def getNumberOfClasses(self):
		raise NotImplementedError("Should have implemented this")

	def getClasses(self):
		raise NotImplementedError("Should have implemented this")
