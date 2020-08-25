from neural_wrappers.readers import DatasetReader, DatasetRange, DatasetIndex
from neural_wrappers.utilities import npCloseEnough
from typing import Any
from overrides import overrides
import numpy as np

class DummyDataset(DatasetReader):
	@overrides
	def getDataset(self, topLevel : str) -> Any:
		return None

	@overrides
	def getNumData(self, topLevel : str) -> int:
		return 0

	@overrides
	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		return DatasetIndex()

class Dataset(DatasetReader):
	def __init__(self):
		dataBuckets = {
			"data" : ["D1", "D2"],
			"labels" : ["D2", "D3"]
		}

		dimGetter = {
			"D1" : lambda dataset, index: dataset["D1"][index.start : index.end],
			"D2" : lambda dataset, index: dataset["D2"][index.start : index.end],
			"D3" : lambda dataset, index: dataset["D3"][index.start : index.end]
		}

		dimTransform = {
			"labels" : {
				"D2" : lambda x : x * 0
			}
		}

		super().__init__(dataBuckets, dimGetter, dimTransform)
		np.random.seed(0)
		self.dataset = {
			"train" : {
				"D1" : np.random.randn(100, 3),
				"D2" : np.random.randn(100, 2),
				"D3" : np.random.randn(100, 5)
			}
		}

	def getDataset(self, topLevel : str) -> Any:
		return self.dataset[topLevel]

	def getNumData(self, topLevel : str) -> int:
		return 100

	def getBatchDatasetIndex(self, i : int, topLevel : str, batchSize : int) -> DatasetIndex:
		startIndex = i * batchSize
		endIndex = min((i + 1) * batchSize, self.getNumData(topLevel))
		assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
		return DatasetRange(startIndex, endIndex)

class TestDatasetReader:
	def test_constructor_1(self):
		reader = DummyDataset(
			dataBuckets = {
				"data" : ["rgb", "depth"],
				"labels" : ["depth", "semantic"]
			},
			dimGetter = {
				"rgb" : lambda dataset, index: None,
				"depth" : lambda dataset, index: None,
				"semantic" : lambda dataset, index: None
			},
			dimTransform = {
				"data" : {
					"rgb" : lambda x : x,
					"depth" : lambda x : x
				},
				"labels" : {
					"depth" : lambda x : x,
					"semantic" : lambda x : x
				}
			}
		)

	def test_constructor_2(self):
		reader = DummyDataset(
			dataBuckets = {
				"data" : ["rgb", "depth"],
				"labels" : ["depth", "semantic"]
			},
			dimGetter = {
				"rgb" : lambda dataset, index: None,
				"depth" : lambda dataset, index: None,
				"semantic" : lambda dataset, index: None
			},
			dimTransform = {
				"data" : {
					"rgb" : lambda x : x,
				},
				"labels" : {
					"semantic" : lambda x : x
				}
			}
		)

	def test_iterate_1(self):
		reader = Dataset()
		steps = reader.getNumIterations("train", 9)
		assert steps == 12
		generator = reader.iterate("train", 9)
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

def main():
	TestDatasetReader().test_constructor_1()
	# TestDatasetReader().test_constructor_2()
	# TestDatasetReader().test_iterate_1()
	pass

if __name__ == "__main__":
	main()