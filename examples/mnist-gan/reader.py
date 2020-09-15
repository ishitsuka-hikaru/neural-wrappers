import numpy as np
from neural_wrappers.readers import MNISTReader, DatasetReader
from functools import partial

# For some reasons, results are much better if provided data is in range -1 : 1 (not 0 : 1 or standardized).
class GANReader(MNISTReader):
	def __init__(self, datasetPath:str, noiseSize:int):
		self.noiseSize = noiseSize
		self.datasetPath = datasetPath

		DatasetReader.__init__(self,
			dataBuckets = {"data" : ["rgb"]},
			dimGetter = {"rgb" : \
				lambda dataset, index : dataset["images"][index.start : index.end]()},
			dimTransform = {
				"data" : {"rgb" : lambda x : (np.float32(x) / 255 - 0.5) * 2}
			}
		)

	def iterate_once(self, type, miniBatchSize):
		for data, labels in super().iterate_once(type, miniBatchSize):
			# We need to give items for the entire training epoch, which includes optimizing both the generator and
			#  the discriminator. We also don't need any labels for the generator, so we'll just pass None
			breakpoint()

			MB = data.shape[0]
			gNoise = np.random.randn(MB, self.noiseSize).astype(np.float32)
			dNoise = np.random.randn(MB, self.noiseSize).astype(np.float32)

			# Pack the data in two components, one for G and one for D
			yield (gNoise, None), ((data, dNoise), None)
