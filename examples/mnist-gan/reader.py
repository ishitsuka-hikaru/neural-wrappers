import numpy as np
from neural_wrappers.readers import H5BatchedDatasetReader
from functools import partial
from overrides import overrides

# For some reasons, results are much better if provided data is in range -1 : 1 (not 0 : 1 or standardized).
class GANReader(H5BatchedDatasetReader):
	def __init__(self, datasetPath:str, latentSpaceSize:int):
		super().__init__(
			datasetPath,
			dataBuckets = {"data" : ["rgb"]},
			dimGetter = {"rgb" : \
				lambda dataset, index : dataset["images"][index.start : index.stop]},
			dimTransform = {
				"data" : {"rgb" : lambda x : (np.float32(x) / 255 - 0.5) * 2}
			}
		)
		self.latentSpaceSize = latentSpaceSize

	@overrides
	def __len__(self) -> int:
		return len(self.getDataset()["images"])

	def __getitem__(self, index):
		item = super().__getitem__(index)
		MB = index.stop - index.start
		return np.random.randn(MB, self.latentSpaceSize).astype(np.float32), item["data"]["rgb"]
