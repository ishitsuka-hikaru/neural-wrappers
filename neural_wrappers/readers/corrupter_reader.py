from .dataset_reader import DatasetReader
import numpy as np

# @brief Generic corruption reader. Only works for float32 data.
# @param[in] baseReader The reader whose values are getting corrupted
# @param[in] corruptionPercent A value in range (0, 100), represeting the probability of each item to get corrupted
#  from the base reader
# @param[in] rangeValues A range (low, high) of values from which the noise is generated. If discreteInterval is True,
#  this parameter can be a list of values from which the noise is generated using a uniform probability
# @param[in]
class CorrupterReader(DatasetReader):
	def __init__(self, baseReader, corruptionPercent, rangeValues, discreteInterval=False):
		assert corruptionPercent > 0 and corruptionPercent < 100
		self.baseReader = baseReader
		self.corruptionPercent = corruptionPercent
		self.sampleProbability = [(100 - corruptionPercent) / 100, corruptionPercent / 100]
		if discreteInterval == False:
			assert len(rangeValues) == 2, "Expected (low, high) pair for noise generation."
			Min, Max = rangeValues
			self.sampler = lambda size : (np.random.choice([Min, Max], p=[0.5, 0.5], size=size) * (Max - Min) + Min)
		else:
			p = np.ones(len(rangeValues)) / len(rangeValues)
			self.sampler = lambda size : np.random.choice(rangeValues, p=p, size=size)

		self.numData = self.baseReader.numData
		self.transforms = self.baseReader.transforms

	def noiseUp(self, inputs):
		mask = np.random.choice([0, 1], p=self.sampleProbability, size=inputs.shape)
		noiseValues = self.sampler(size=inputs.shape)

		# mask = 0 => inputs * 1 + noiseValues * 0 (no change)
		# mask = 1 => inputs * 0 + noiseValues * 1 (change value at that place)
		newInputs = (inputs * (1 - mask) + noiseValues * mask).astype(inputs.dtype)
		return newInputs

	def iterate_once(self, type, miniBatchSize):
		baseGenerator = self.baseReader.iterate_once(type, miniBatchSize)

		for inputs, _ in baseGenerator:
			yield self.noiseUp(inputs), inputs