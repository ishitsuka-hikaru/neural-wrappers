import numpy as np
from overrides import overrides
from functools import partial
from neural_wrappers.readers import H5BatchedDatasetReader
from neural_wrappers.readers.batched_dataset_reader.h5_batched_dataset_reader import defaultH5DimGetter
from neural_wrappers.utilities import toCategorical

def topLeftFn(x):
	x = np.float32(x) / 255
	x[:, 0:14, 0:14] = 0
	return x

def topRightFn(x):
	x = np.float32(x) / 255
	x[:, 0:14, 14:] = 0
	return x

def bottomLeftFn(x):
	x = np.float32(x) / 255
	x[:, 14:, 0:14] = 0
	return x

def bottomRightFn(x):
	x = np.float32(x) / 255
	x[:, 14:, 14:] = 0
	return x

class Reader(H5BatchedDatasetReader):
	def __init__(self, datasetPath:str, normalization:str = "min_max_0_1"):
		assert normalization in ("none", "min_max_0_1")

		getterFn = partial(defaultH5DimGetter, dim="images")
		super().__init__(datasetPath,
			dataBuckets = {
				"data" : ["rgb", "rgb_top_left", "rgb_top_right", "rgb_bottom_left", "rgb_bottom_right", "labels"],
			},
			dimGetter = {
				"rgb" : getterFn,
				"rgb_top_left" : getterFn,
				"rgb_top_right" : getterFn,
				"rgb_bottom_left" : getterFn,
				"rgb_bottom_right" : getterFn
			},
			dimTransform = {
				"data" : {
					"rgb" : lambda x : np.float32(x) / 255,
					"rgb_top_left" : topLeftFn,
					"rgb_top_right" : topRightFn,
					"rgb_bottom_left" : bottomLeftFn,
					"rgb_bottom_right" : bottomRightFn,
					"labels" : lambda x : toCategorical(x, numClasses=10)
				},
			}
		)

	@overrides
	def __len__(self) -> int:
		return len(self.getDataset()["images"])

	@overrides
	def __getitem__(self, key):
		item = super().__getitem__(key)
		return item["data"], item["data"]
