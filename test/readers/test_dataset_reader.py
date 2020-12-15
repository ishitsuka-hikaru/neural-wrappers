import numpy as np
from overrides import overrides
from typing import Any, Iterator, Dict
from neural_wrappers.readers import DatasetReader, DatasetRange, DatasetIndex

class DummyDataset(DatasetReader):
	@overrides
	def getDataset(self, topLevel:str) -> Any:
		return None

	@overrides
	def getNumData(self, topLevel:str) -> int:
		return 0

	@overrides
	def getNumIterations(self, topLevel:str) -> int:
		return 0

	@overrides
	def getIndex(self, topLevel:str, i:int) -> DatasetIndex:
		return DatasetIndex()

	@overrides
	def iterateOneEpoch(self, topLevel) -> Iterator[Dict[str, np.ndarray]]:
		pass

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

def main():
	TestDatasetReader().test_constructor_1()
	# TestDatasetReader().test_constructor_2()
	# TestDatasetReader().test_iterate_1()
	pass

if __name__ == "__main__":
	main()