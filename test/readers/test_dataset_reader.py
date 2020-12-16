import numpy as np
from overrides import overrides
from typing import Any, Iterator, Dict, Callable
from neural_wrappers.readers import DatasetReader, DatasetIndex

class DummyDataset(DatasetReader):
	@overrides
	def getDataset(self) -> Any:
		return None

	@overrides
	def getNumData(self) -> int:
		return 0

	@overrides
	def getNumData(self) -> int:
		return 0

	@overrides
	def getItem(self, i):
		pass

class TestDatasetReader:
	dataBuckets = {
		"data" : ["rgb", "depth"],
		"labels" : ["depth", "semantic"]
	}
	dimGetter = {
		"rgb" : lambda dataset, index: None,
		"depth" : lambda dataset, index: None,
		"semantic" : lambda dataset, index: None
	}
	dimTransform = {
		"data" : {
			"rgb" : lambda x : x,
			"depth" : lambda x : x
		},
		"labels" : {
			"depth" : lambda x : x,
		}
	}

	def test_constructor_1(self):
		reader = DummyDataset(dataBuckets=TestDatasetReader.dataBuckets, \
			dimGetter=TestDatasetReader.dimGetter, dimTransform=TestDatasetReader.dimTransform)
		print(reader)
		assert not reader is None
		assert reader.datasetFormat.dataBuckets == {"data" : ["rgb", "depth"], "labels" : ["depth", "semantic"]}
		assert list(reader.datasetFormat.dimGetter.keys()) == ["rgb", "depth", "semantic"]
		for v in reader.datasetFormat.dimGetter.values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform.keys()) == ["data", "labels"]
		assert list(reader.datasetFormat.dimTransform["data"].keys()) == ["rgb", "depth"]
		for v in reader.datasetFormat.dimTransform["data"].values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform["labels"].keys()) == ["depth", "semantic"]
		for v in reader.datasetFormat.dimTransform["labels"].values():
			assert isinstance(v, Callable)

def main():
	TestDatasetReader().test_constructor_1()

if __name__ == "__main__":
	main()