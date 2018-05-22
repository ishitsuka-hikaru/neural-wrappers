from .dataset_reader import DatasetReader
from neural_wrappers.utilities import isBaseOf
import numpy as np

# @brief Generic class that combines two or more dataset readers
# @param[in] readers A list of two or more instantiated dataset readers.
class CombinedDatasetReader(DatasetReader):
	def __init__(self, readers):
		assert len(readers) > 1, "Expected at least two instantiated dataset readers"
		self.readers = readers
		self.setup()

	def setup(self):
		assert isBaseOf(self.readers[0], DatasetReader), \
			"Expected type of reader to be a base of DatasetReader, got %s at index %0" % (type(reader))
		for i, reader in enumerate(self.readers[1 :]):
			assert isBaseOf(reader, DatasetReader), \
				"Expected type of reader to be a base of DatasetReader, got %s at index %d" % (type(reader), i)
			assert reader.dataShape == self.readers[0].dataShape
			assert reader.labelShape == self.readers[0].labelShape

	# @brief Compute the contribution of each dataset reader according to how much iterations each would do. We try
	#  to make them do the same amount of iterations (or as close enough as possible), by altering the mini batch of
	#  each reader.
	def getActualMiniBatchSize(self, type, miniBatchSize, accountTransforms):
		numIterations = [reader.getNumIterations(type, miniBatchSize, accountTransforms) for reader in self.readers]
		numIterations = np.array(numIterations)
		assert np.sum(numIterations == 0) == 0, "The minibatch is too small, and some readers have a batch size of 0"

		totalNumIterations = np.sum(numIterations)
		percentEachReader = numIterations / totalNumIterations

		# Each reader gets a mini batch size according to the percentage, but sinc it must be an integer, we must cast
		#  them. Thus, the last reader will get a (potential) higher contribution, which is the initial value minus
		#  the sum of all the others. If everything divides properly, that value is simply it's normal contribution,
		#  otherewise it's a little higher and last iterations will only uses items from that reader.
		actualMiniBatch = np.int32(percentEachReader * miniBatchSize)
		actualMiniBatch[-1] = miniBatchSize - np.sum(actualMiniBatch[0 : -1])
		return actualMiniBatch

	def getNumIterations(self, type, miniBatchSize, accountTransforms=True):
		miniBatches = self.getActualMiniBatchSize(type, miniBatchSize, accountTransforms)

		numIterations = np.array([self.readers[i].getNumIterations(type, miniBatches[i], accountTransforms) \
			for i in range(len(self.readers))])
		# Since values are truncated to floor integers, the number of iterations will be slightly different. The
		#  steps between max and min value will only use items from the dataset that holds the max value.
		return np.max(numIterations)

	def iterate_once(self, type, miniBatchSize):
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=True)
		miniBatchSizes = self.getActualMiniBatchSize(type, miniBatchSize, accountTransforms=True)
		generators = [self.readers[i].iterate_once(type, miniBatchSizes[i]) for i in range(len(self.readers))]
		numGenerators = len(generators)

		for i in range(numIterations):
			data, labels = [], []
			for j in range(numGenerators):
				try:
					readerData, readerLabels = next(generators[j])
					data.append(readerData)
					labels.append(readerLabels)
				except StopIteration:
					pass
			data = np.concatenate(data, axis=0)
			labels = np.concatenate(labels, axis=0)
			yield data, labels