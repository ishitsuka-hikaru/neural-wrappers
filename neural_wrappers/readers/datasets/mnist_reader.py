import numpy as np
from functools import partial
from overrides import overrides
from typing import Iterator, Tuple
from ..h5_batched_dataset_reader import H5BatchedDatasetReader
from ...utilities import toCategorical

class MNISTReader(H5BatchedDatasetReader):
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

	def setBatchSize(self, batchSize:int):
		N = self.getNumData()
		n = N // batchSize
		batchSizes = n * [batchSize]
		if N % batchSize != 0:
			batchSizes.append(N % batchSize)
		self.batchSizes = batchSizes

	@overrides
	def getBatchSizes(self):
		return self.batchSizes
