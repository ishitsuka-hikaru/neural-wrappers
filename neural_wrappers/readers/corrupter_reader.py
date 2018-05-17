from .dataset_reader import DatasetReader
import numpy as np

class CorrupterReader(DatasetReader):
	def __init__(self, baseReader, corruptionPercent):
		assert corruptionPercent > 0 and corruptionPercent < 100
		self.baseReader = baseReader
		self.corruptionPercent = corruptionPercent

		self.sampleProbability = [(100 - corruptionPercent) / 100, corruptionPercent / 100]
		self.numData = self.baseReader.numData
		self.transforms = self.baseReader.transforms

	def noiseUp(self, inputs, Min, Max):
		mask = np.random.choice([0, 1], p=self.sampleProbability, size=inputs.shape)
		noiseValues = np.random.choice([Min, Max], p=[0.5, 0.5], size=inputs.shape)

		# mask = 0 => inputs * 1 + noiseValues * 0 (no change)
		# mask = 1 => inputs * 0 + noiseValues * 1 (change value at that place)		
		newInputs = inputs * (1 - mask) + noiseValues * mask
		return newInputs

	def iterate_once(self, type, miniBatchSize):
		baseGenerator = self.baseReader.iterate_once(type, miniBatchSize)

		# Do first time, to establish min/max values (which are used for noising)
		inputs, _ = next(baseGenerator)
		Min, Max = np.min(inputs), np.max(inputs)
		yield self.noiseUp(inputs, Min, Max), inputs

		for inputs, _ in baseGenerator:
			yield self.noiseUp(inputs, Min, Max), inputs