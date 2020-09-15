import numpy as np
from neural_wrappers.readers import H5DatasetReader
from functools import partial
from overrides import overrides

# For some reasons, results are much better if provided data is in range -1 : 1 (not 0 : 1 or standardized).
class GANReader(H5DatasetReader):
	def __init__(self, datasetPath:str, noiseSize:int):
		self.noiseSize = noiseSize
		self.datasetPath = datasetPath

		super().__init__(
			datasetPath,
			dataBuckets = {"data" : ["rgb"]},
			dimGetter = {"rgb" : \
				lambda dataset, index : dataset["images"][index.start : index.end]},
			dimTransform = {
				"data" : {"rgb" : lambda x : (np.float32(x) / 255 - 0.5) * 2}
			}
		)

	@overrides
	def iterateOneEpoch(self, topLevel : str, batchSize : int):
		for items in super().iterateOneEpoch(topLevel, batchSize):
			rgb = items["data"]["rgb"]
			MB = len(rgb)
			fakeGenerator = np.random.randn(MB, self.noiseSize).astype(np.float32)
			fakeDiscriminator = np.random.randn(MB, self.noiseSize).astype(np.float32)
			data = {
				"real" : rgb,
				"fakeGenerator" : fakeGenerator,
				"fakeDiscriminator" : fakeDiscriminator
			}
			yield data, data

	@overrides
	def getNumData(self, topLevel : str) -> int:
		return len(self.dataset[topLevel]["images"])
