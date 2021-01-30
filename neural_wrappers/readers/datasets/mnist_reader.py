import numpy as np
from functools import partial
from overrides import overrides
from typing import Iterator, Tuple, List
from ..batched_dataset_reader import H5BatchedDatasetReader
from ..batched_dataset_reader.utils import batchIndexFromBatchSizes
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

	@overrides
	def __len__(self) -> int:
		return len(self.getDataset()["images"])

	@overrides
	def __getitem__(self, index):
		item, B = super().__getitem__(index)
		return (item["data"], item["labels"]["labels"]), B
