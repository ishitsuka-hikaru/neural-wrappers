import numpy as np
from neural_wrappers.readers import MNISTReader

class GANReader(MNISTReader):
	def __init__(self, noiseSize, **kwargs):
		self.noiseSize = noiseSize
		super().__init__(**kwargs)

	def iterate_once(self, type, miniBatchSize):
		for data, labels in super().iterate_once(type, miniBatchSize):
			# We need to give items for the entire training epoch, which includes optimizing both the generator and
			#  the discriminator. We also don't need any labels for the generator, so we'll just pass None

			MB = data.shape[0]
			gNoise = np.random.randn(MB, self.noiseSize).astype(np.float32)
			dNoise = np.random.randn(MB, self.noiseSize).astype(np.float32)

			# Pack the data in two components, one for G and one for D
			yield (gNoise, None), ((data, dNoise), None)
