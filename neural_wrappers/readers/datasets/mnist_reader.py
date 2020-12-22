import numpy as np
from functools import partial
from overrides import overrides
from typing import Iterator, Tuple, List
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
		self.batches = []
		self.batchSize = 0
		self.isCacheable = True

	# @param[in] batchSize The static batch size required to iterate one epoch. If the batch size is not divisible by
	#  the number of items, the last batch will trimmed accordingly. If the provided value is -1, it is set to the
	#  default value of the entire dataset, based on self.getNumData()
	def setBatchSize(self, batchSize:int):
		assert batchSize == 1 or batchSize > 0
		N = len(self)
		if batchSize == -1:
			batchSize = N
		n = N // batchSize
		batches = n * [batchSize]
		if N % batchSize != 0:
			batches.append(N % batchSize)
		self.batchSize = batchSize
		self.batches = batches

	@overrides
	def getBatches(self) -> List[int]:
		return self.batches

	@overrides
	def getNumData(self) -> int:
		return len(self.getDataset()["images"])