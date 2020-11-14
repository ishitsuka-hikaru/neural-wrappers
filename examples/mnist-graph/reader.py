import numpy as np
from overrides import overrides
from neural_wrappers.readers import H5DatasetReader
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

class Reader(H5DatasetReader):
	def __init__(self, datasetPath:str):
		super().__init__(datasetPath,
			dataBuckets = {
				"data" : ["rgb", "rgb_top_left", "rgb_top_right", "rgb_bottom_left", "rgb_bottom_right", "label"],
			},
			dimGetter = {
				"rgb" : lambda dataset, index : dataset["images"][index.start : index.end][()],
				"rgb_top_left" : lambda dataset, index : dataset["images"][index.start : index.end][()],
				"rgb_top_right" : lambda dataset, index : dataset["images"][index.start : index.end][()],
				"rgb_bottom_left" : lambda dataset, index : dataset["images"][index.start : index.end][()],
				"rgb_bottom_right" : lambda dataset, index : dataset["images"][index.start : index.end][()],
				"label" : lambda dataset, index : dataset["labels"][index.start : index.end][()],
			},
			dimTransform = {
				"data" : {
					"rgb" : lambda x : np.float32(x) / 255,
					"rgb_top_left" : topLeftFn,
					"rgb_top_right" : topRightFn,
					"rgb_bottom_left" : bottomLeftFn,
					"rgb_bottom_right" : bottomRightFn,
					"label" : lambda x : toCategorical(x, numClasses=10)
				},
			}
		)

	@overrides
	def getNumData(self, topLevel:str) -> int:
		return {
			"train" : 60000,
			"test" : 10000
		}[topLevel]

	@overrides
	def iterateOneEpoch(self, topLevel:str, batchSize:int):
		for items in super().iterateOneEpoch(topLevel, batchSize):
			yield items["data"], items["data"]