import numpy as np
from functools import partial
from overrides import overrides
from ..h5_dataset_reader import H5DatasetReader, defaultH5DimGetter
from ...utilities import toCategorical
from typing import Iterator, Tuple

class MNISTReader(H5DatasetReader):
	def __init__(self, datasetPath:str, normalization:str = "min_max_0_1"):
		assert normalization in ("none", "min_max_0_1")

		rgbTransform = {
			"min_max_0_1" : lambda x : np.float32(x) / 255,
			"none" : lambda x : x
		}[normalization]

		super().__init__(datasetPath,
			dataBuckets = {"data" : ["rgb"], "labels" : ["labels"]},
			dimGetter = {"rgb" : partial(defaultH5DimGetter, dim="images")},
			dimTransform = {
				"data" : {"rgb" : rgbTransform},
				"labels" : {"labels" : lambda x : toCategorical(x, numClasses=10)}
			}
		)

	@overrides
	def getNumData(self, topLevel:str) -> int:
		return {
			"train" : 60000,
			"test" : 10000
		}[topLevel]

	@overrides
	def iterateOneEpoch(self, topLevel:str, batchSize:int) -> Iterator[Tuple[str, np.ndarray]]:
		for items in super().iterateOneEpoch(topLevel, batchSize):
			yield items["data"]["rgb"], items["labels"]["labels"]
