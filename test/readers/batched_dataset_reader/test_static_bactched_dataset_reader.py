import numpy as np
from typing import Any
from overrides import overrides
from neural_wrappers.readers import StaticBatchedDatasetReader, DatasetRange, DatasetIndex
from neural_wrappers.utilities import npCloseEnough

class Dataset(StaticBatchedDatasetReader):
	def __init__(self, batchSize):
		dataBuckets = {
			"data" : ["D1", "D2"],
			"labels" : ["D2", "D3"]
		}

		dimGetter = {
			"D1" : lambda dataset, index: dataset["D1"][index.start : index.stop],
			"D2" : lambda dataset, index: dataset["D2"][index.start : index.stop],
			"D3" : lambda dataset, index: dataset["D3"][index.start : index.stop]
		}

		dimTransform = {
			"labels" : {
				"D2" : lambda x : x * 0
			}
		}

		super().__init__(dataBuckets, dimGetter, dimTransform, batchSize)
		np.random.seed(0)
		self.dataset = {
			"train" : {
				"D1" : np.random.randn(100, 3),
				"D2" : np.random.randn(100, 2),
				"D3" : np.random.randn(100, 5)
			}
		}

	@overrides
	def getDataset(self, topLevel:str) -> Any:
		return self.dataset[topLevel]

	@overrides
	def getNumData(self, topLevel:str) -> int:
		return 100

class TestBatchedDatasetReader:
	def test_iterate_1(self):
		reader = Dataset(batchSize=9)
		steps = reader.getNumIterations("train")
		assert steps == 12
		generator = reader.iterateOneEpoch("train")
		assert not generator is None
		items = next(generator)
		assert not items is None
		assert list(items.keys()) == ["data", "labels"]
		assert list(items["data"]) == ["D1", "D2"]
		assert list(items["labels"]) == ["D2", "D3"]

		assert npCloseEnough(items["data"]["D1"], reader.dataset["train"]["D1"][0 : 9])
		assert npCloseEnough(items["data"]["D2"], reader.dataset["train"]["D2"][0 : 9])
		assert npCloseEnough(items["labels"]["D3"], reader.dataset["train"]["D3"][0 : 9])
		assert items["labels"]["D2"].sum() == 0