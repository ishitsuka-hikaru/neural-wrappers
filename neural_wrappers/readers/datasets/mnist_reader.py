import numpy as np
from functools import partial
from overrides import overrides
from typing import Iterator, Tuple
from ..batched_reader import H5DatasetReader
from ..batched_reader.h5_dataset_reader import defaultH5DimGetter
from ...utilities import toCategorical

class MNISTReader(H5DatasetReader):
	def __init__(self, datasetPath:str, normalization:str = "min_max_0_1"):
		assert normalization in ("none", "min_max_0_1")

		rgbTransform = {
			"min_max_0_1" : (lambda x : np.float32(x) / 255),
			"none" : (lambda x : x)
		}[normalization]

		super().__init__(datasetPath,
			dataBuckets = {"data" : ["images"], "labels" : ["labels"]},
			dimTransform = {
				"data" : {"images" : rgbTransform},
				"labels" : {"labels" : lambda x : toCategorical(x, numClasses=10)}
			}
		)

	def setBatchSize(self, batchSize):
		self.batchSize = batchSize

	@overrides
	def getNumIterations(self, topLevel:str) -> int:
		N = self.getNumData(topLevel)
		B = self.getBatchSize()
		n = N // B + (N % B != 0)
		return n

	@overrides
	def getBatchSize(self, topLevel=None, i=None):
		return self.batchSize

	@overrides
	def iterateOneEpoch(self, topLevel:str) -> Iterator[Tuple[str, np.ndarray]]:
		for items in super().iterateOneEpoch(topLevel):
			yield items["data"]["images"], items["labels"]["labels"]
